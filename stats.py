import pandas as pd
import os
from tqdm import tqdm 

def get_subject_wise_acc(file,
                        subject_column="subject_name",
                        actual_column="cop",
                        prediction_column="predictions"):
    
    def acc(group_df):
        return len(group_df[group_df[actual_column]==group_df[prediction_column]])/len(group_df)

    df = pd.read_csv(file)
    grouped = df.groupby(by=[subject_column])
    file_type = os.path.basename(file).split("_")[0]
    grouped.apply(acc).to_frame("acc").to_csv(os.path.join(os.path.dirname(file),f"{file_type}_subj_wise_acc.csv"))

def subject_wise_acc_report(parent_models_folder):
    for model_folder in tqdm(os.listdir(parent_models_folder)):
        dev_file = os.path.join(parent_models_folder,model_folder,"dev_results.csv")
        test_file = os.path.join(parent_models_folder,model_folder,"test_results.csv")
        if os.path.exists(dev_file):
            get_subject_wise_acc(dev_file)
        if os.path.exists(test_file):
            get_subject_wise_acc(test_file)


if __name__ == "__main__":
    subject_wise_acc_report("/home/models")


