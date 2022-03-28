from torch.utils.data import Dataset
import pandas as pd

class MCQADataset(Dataset):

  def __init__(self,
               csv_path,
               use_context=True):
#     self.dataset = dataset['train'] if training == True else dataset['test']
    self.dataset = pd.read_csv(csv_path)
    self.use_context = use_context

  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self,idx):
    return_tuple = tuple()
    if self.use_context:
      context = self.dataset.loc[idx,'exp']
      return_tuple+=(context,)
    question = self.dataset.loc[idx,'question']
    options = self.dataset.loc[idx,['opa', 'opb', 'opc', 'opd']].values
    label = self.dataset.loc[idx,'cop'] - 1
    return_tuple+=(question,options,label)
    return return_tuple