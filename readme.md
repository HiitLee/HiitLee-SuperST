### SuperST: Superficial Self-Training for Few-Shot Text Classification
In few-shot text classification, self-training is a popular tool in semi-supervised learning (SSL). It relies on pseudo-labels to expand data, which has demonstrated success. However, these pseudo-labels contain potential noise and provoke a risk of underfitting the decision boundary. While the pseudo-labeled data can indeed be noisy, fully acquiring this flawed data can result in the accumulation of further noise and eventually impacting the model performance. Consequently, self-training presents a challenge: mitigating the accumulation of noise in the pseudo-labels. Confronting this challenge, we introduce superficial learning, inspired by pedagogy’s focus on essential knowledge. Superficial learning in pedagogy is a learning scheme that only learns the material ‘at some extent’, not fully understanding the material. This approach is usually avoided in education but counter-intuitively in our context, we employ superficial learning to acquire only the necessary context from noisy data, effectively avoiding the noise. This concept serves as the foundation for SuperST, our self-training framework. SuperST applies superficial learning to the noisy data and fine-tuning to the less noisy data, creating an efficient learning cycle that prevents overfitting to the noise and spans the decision boundary effectively. Notably, SuperST improves the classifier accuracy for few-shot text classification by 18.5% at most and 8.0% in average, compared with the state-of-the-art SSL baselines. We substantiate our claim through empirical experiments and decision boundary analysis 

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


