import pytorch_lightning as pl
from pytorch_lightning.core.step_result import TrainResult,EvalResult
from pytorch_lightning import Trainer
from torch.utils.data import SequentialSampler,RandomSampler
from torch import nn
import numpy as np
import math
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader,RandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer,AutoModel
import functools

class MCQAModel(pl.LightningModule):
  def __init__(self,
               model_name_or_path,
               args):
    
    super().__init__()
    self.init_encoder_model(model_name_or_path)
    self.args = args
    self.batch_size = self.args['batch_size']
    self.dropout = nn.Dropout(self.args['hidden_dropout_prob'])
    self.linear = nn.Linear(in_features=self.args['hidden_size'],out_features=1)
    self.ce_loss = nn.CrossEntropyLoss()
    self.save_hyperparameters()
    
  
  def init_encoder_model(self,model_name_or_path):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    self.model = AutoModel.from_pretrained(model_name_or_path)
 
  def prepare_dataset(self,train_dataset,val_dataset,test_dataset=None):
    """
    helper to set the train and val dataset. Doing it during class initialization
    causes issues while loading checkpoint as the dataset class needs to be 
    present for the weights to be loaded.
    """
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    if test_dataset != None:
        self.test_dataset = test_dataset
    else:
        self.test_dataset = val_dataset
  
  def forward(self,input_ids,attention_mask,token_type_ids):
    outputs = self.model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
    
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)
    logits = self.linear(pooled_output)
    reshaped_logits = logits.view(-1,self.args['num_choices'])
    return reshaped_logits
  
  def training_step(self,batch,batch_idx):
    inputs,labels = batch
    for key in inputs:
      inputs[key] = inputs[key].to(self.args["device"])
    logits = self(**inputs)
    loss = self.ce_loss(logits,labels)
    result = TrainResult(loss)
    result.log('train_loss', loss, on_epoch=True)
    return result
  
  def test_step(self, batch, batch_idx):
    inputs,labels = batch
    for key in inputs:
      inputs[key] = inputs[key].to(self.args["device"])
    logits = self(**inputs)
    loss = self.ce_loss(logits,labels)
    result = EvalResult(loss)
    result.log('test_loss', loss, on_epoch=True)
    result.log('logits',logits,on_epoch=True)
    result.log('labels',labels,on_epoch=True)
    self.log('test_loss', loss)
    return result
 
  def test_epoch_end(self, outputs):
    avg_loss = outputs['test_loss'].mean()
    predictions = torch.argmax(outputs['logits'],axis=-1)
    labels = outputs['labels']
    self.test_predictions = predictions
    correct_predictions = torch.sum(predictions==labels)
    accuracy = correct_predictions.cpu().detach().numpy()/predictions.size()[0]
    result = EvalResult(checkpoint_on=avg_loss,early_stop_on=avg_loss)
    result.log_dict({"test_loss":avg_loss,"test_acc":accuracy},prog_bar=True,on_epoch=True)
    self.log('avg_test_loss', avg_loss)
    self.log('avg_test_acc', accuracy)
    return result
  
  def validation_step(self, batch, batch_idx):
    inputs,labels = batch
    for key in inputs:
      inputs[key] = inputs[key].to(self.args["device"])
    logits = self(**inputs)
    loss = self.ce_loss(logits,labels)
    result = EvalResult(loss)
    result.log('val_loss', loss, on_epoch=True)
    result.log('logits',logits,on_epoch=True)
    result.log('labels',labels,on_epoch=True)
    self.log('val_loss', loss)
    return result

  def validation_epoch_end(self, outputs):
        avg_loss = outputs['val_loss'].mean()
        predictions = torch.argmax(outputs['logits'],axis=-1)
        labels = outputs['labels']
        correct_predictions = torch.sum(predictions==labels)
        accuracy = correct_predictions.cpu().detach().numpy()/predictions.size()[0]
        result = EvalResult(checkpoint_on=avg_loss,early_stop_on=avg_loss)
        result.log_dict({"val_loss":avg_loss,"val_acc":accuracy},prog_bar=True,on_epoch=True)
        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_acc', accuracy)
        return result
        
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(),lr=self.args['learning_rate'],eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=(self.args['num_epochs'] + 1) * math.ceil(len(self.train_dataset) / self.args['batch_size']),
    )
    return [optimizer],[scheduler]
  
  def process_batch(self,batch,tokenizer,max_len=32):
    expanded_batch = []
    labels = []
    context = None
    for data_tuple in batch:
        if len(data_tuple) == 4:
          context,question,options,label = data_tuple
        else:
          question,options,label = data_tuple
        question_option_pairs = [question+' '+option for option in options]
        labels.append(label)

        if context:
          contexts = [context]*len(options)
          expanded_batch.extend(zip(contexts,question_option_pairs))
        else:
          expanded_batch.extend(question_option_pairs)
    tokenized_batch = tokenizer.batch_encode_plus(expanded_batch,truncation=True,padding="max_length",max_length=max_len,return_tensors="pt")

    return tokenized_batch,torch.tensor(labels)
  
  def train_dataloader(self):
    train_sampler = RandomSampler(self.train_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
      )
    train_dataloader = DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                sampler=train_sampler,
                                collate_fn=model_collate_fn)
    return train_dataloader
  
  def val_dataloader(self):
    eval_sampler = SequentialSampler(self.val_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
      )
    val_dataloader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                sampler=eval_sampler,
                                collate_fn=model_collate_fn)
    return val_dataloader
  
  def test_dataloader(self):
    eval_sampler = SequentialSampler(self.test_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
      )
    test_dataloader = DataLoader(self.test_dataset,
                                batch_size=self.batch_size,
                                sampler=eval_sampler,
                                collate_fn=model_collate_fn)
    return test_dataloader
