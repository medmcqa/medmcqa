from dataclasses import dataclass


@dataclass
class Arguments:
    train_csv:str
    test_csv:str 
    dev_csv:str
    batch_size:int = 16
    max_len:int = 192
    checkpoint_batch_size:int = 32
    print_freq:int = 100
    pretrained_model_name:str = "bert-base-uncased"
    learning_rate:float = 2e-4
    hidden_dropout_prob:float =0.4
    hidden_size:int=768
    num_epochs:int = 5
    num_choices:int = 4
    device:str='cuda'
    gpu='0'
    use_context:bool=True