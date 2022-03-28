for model in microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext #allenai/scibert_scivocab_uncased bert-base-uncased 
do
    for datafolder in all_csv
    do
       python3 train.py --model $model --dataset_folder_name $datafolder --use_context
    done
done