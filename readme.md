### Requirements

    python >= 3.6
    
    fire, torch, numpy, tqdm, tensorflow, sklearn, transformers
    
    pip install -r requirements.txt

### Download BERT

Please download pre-trained model **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)** on https://github.com/google-research/bert#pre-trained-models.

### Datasets

We use four benchmark datasets. (IMDB review, AG News, Yahoo! answer, DBpedia). We randomly sample {5, 10, 200, 2500} sentences per class of the original training set as labeled set and {5, 10, 2000} sentences per class of the original training set as validation set. We also randomly sample 5000 - 30000 sentences per class of the original training set as unlabeled set and remove the labels of the labeled set. All data have a balanced class distribution. 


### Training and evaluation of the SuperST(BERT)

Before running this example you must download the **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)** and unzip it to some directory `$model_BERT/ `.

Before running this example you must download the total_data.zip and unzip it to some directory `$data/ `.

    sh train_eval_BERT_IMDB.sh (IMDB)
    sh train_eval_BERT_AGNews.sh (AGNews)
    sh train_eval_BERT_yahoo.sh (Yahoo! answer)
    sh train_eval_BERT_DBpedia.sh (DBpedia)


