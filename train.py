from conf.args import Arguments
from model import MCQAModel
from dataset import MCQADataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.core.step_result import TrainResult,EvalResult
from pytorch_lightning import Trainer
import torch,os,sys
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process
import time,argparse


EXPERIMENT_DATASET_FOLDER = "/home/all_csvs"
WB_PROJECT = "MEDMCQA"

def train(gpu,
          args,
          exp_dataset_folder,
          experiment_name,
          models_folder,
          version):

    pl.seed_everything(42)
    
    EXPERIMENT_FOLDER = os.path.join(models_folder,experiment_name)
    os.makedirs(EXPERIMENT_FOLDER,exist_ok=True)
    experiment_string = experiment_name+'-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'


    wb = WandbLogger(project=WB_PROJECT,name=experiment_name,version=version)
    csv_log = CSVLogger(models_folder, name=experiment_name, version=version)

    train_dataset = MCQADataset(args.train_csv,args.use_context)
    test_dataset = MCQADataset(args.test_csv,args.use_context)
    val_dataset = MCQADataset(args.dev_csv,args.use_context)

    es_callback = pl.callbacks.EarlyStopping(monitor='val_loss',
                                    min_delta=0.00,
                                    patience=2,
                                    verbose=True,
                                    mode='min')

    cp_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                               filepath=os.path.join(EXPERIMENT_FOLDER,experiment_string),
                                               save_top_k=1,
                                               save_weights_only=True,
                                               mode='min')

    mcqaModel = MCQAModel(model_name_or_path=args.pretrained_model_name,
                      args=args.__dict__)
    
    mcqaModel.prepare_dataset(train_dataset=train_dataset,
                              test_dataset=test_dataset,
                              val_dataset=val_dataset)

    trainer = Trainer(gpus=gpu,
                    distributed_backend='ddp' if not isinstance(gpu,list) else None,
                    precision=16,
                    logger=[wb,csv_log],
                    callbacks= [es_callback,cp_callback],
                    max_epochs=args.num_epochs)
    
    trainer.fit(mcqaModel)
    print(f"Training completed")

    ckpt = [f for f in os.listdir(EXPERIMENT_FOLDER) if f.endswith('.ckpt')]

    inference_model = MCQAModel.load_from_checkpoint(os.path.join(EXPERIMENT_FOLDER,ckpt[0]))
    inference_model = inference_model.to("cuda")
    inference_model = inference_model.eval()

    _,test_results = trainer.test(ckpt_path=os.path.join(EXPERIMENT_FOLDER,ckpt[0]))
    wb.log_metrics(test_results)
    csv_log.log_metrics(test_results)


    #Persist test dataset predictions
    test_df = pd.read_csv(args.test_csv)
    test_df.loc[:,"predictions"] = [pred+1 for pred in run_inference(inference_model,mcqaModel.test_dataloader(),args)]
    test_df.to_csv(os.path.join(EXPERIMENT_FOLDER,"test_results.csv"),index=False)
    print(f"Test predictions written to {os.path.join(EXPERIMENT_FOLDER,'test_results.csv')}")

    val_df = pd.read_csv(args.dev_csv)
    val_df.loc[:,"predictions"] = [pred+1 for pred in run_inference(inference_model,mcqaModel.val_dataloader(),args)]
    val_df.to_csv(os.path.join(EXPERIMENT_FOLDER,"dev_results.csv"),index=False)
    print(f"Val predictions written to {os.path.join(EXPERIMENT_FOLDER,'dev_results.csv')}")

    del mcqaModel
    del inference_model
    del trainer
    torch.cuda.empty_cache()
    
def run_inference(model,dataloader,args):
    predictions = []
    for idx,(inputs,labels) in tqdm(enumerate(dataloader)):
        batch_size = len(labels)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction_idxs = torch.argmax(outputs,axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions
        


if __name__ == "__main__":

    models = ["allenai/scibert_scivocab_uncased","bert-base-uncased"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default="bert-base-uncased",help="name of the model")
    parser.add_argument("--dataset_folder_name", default="all_csv",help="dataset folder")
    parser.add_argument("--use_context",default=False,action='store_true',help="mention this flag to use_context")
    cmd_args = parser.parse_args()

    exp_dataset_folder = os.path.join(EXPERIMENT_DATASET_FOLDER,cmd_args.dataset_folder_name)
    model = cmd_args.model
    # if((model == "allenai/scibert_scivocab_uncased" and os.path.basename(exp_dataset_folder) == "single_high_pubmed_exp") or 
    #    (model == "allenai/scibert_scivocab_uncased" and os.path.basename(exp_dataset_folder) == "multi_high_pubmed_exp")):
    #     exit()
    print(f"Training started for model - {model} variant - {exp_dataset_folder} use_context - {str(cmd_args.use_context)}")

    args = Arguments(train_csv=os.path.join(exp_dataset_folder,"train.csv"),
                    test_csv=os.path.join(exp_dataset_folder,"test.csv"),
                    dev_csv=os.path.join(exp_dataset_folder,"dev.csv"),
                    pretrained_model_name=model,
                    use_context=cmd_args.use_context)
    
    exp_name = f"{model}@@@{os.path.basename(exp_dataset_folder)}@@@use_context{str(cmd_args.use_context)}@@@seqlen{str(args.max_len)}".replace("/","_")


    train(gpu=args.gpu,
        args=args,
        exp_dataset_folder=exp_dataset_folder,
        experiment_name=exp_name,
        models_folder="./models",
        version=exp_name)
    
    time.sleep(60)








