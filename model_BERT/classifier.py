import itertools
import csv
import fire
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tokenization
import models
import optim
import train
from utils import set_seeds, get_device, truncate_tokens_pair
import os
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text


def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, topn))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

#################################################################################################################


class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, file, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r", encoding='utf-8') as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t')
            for instance in self.get_instances(lines): # instance : tuple of fields
                for proc in pipeline: # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
        
    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return [tensor[index] for tensor in self.tensors]

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError

class IM(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 0, None): # skip header
            yield line[0], line[1].encode('utf8'),None # label, text_a, text_b

            
class AG(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1", "2", "3") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 0, None): # skip header
            yield line[0], line[1].encode('utf8'),None # label, text_a, text_b


class YA(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1", "2", "3", "4", "5","6", "7", "8", "9") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 0, None): # skip header
            yield line[0], line[1].encode('utf8'),None # label, text_a, text_b

class DB(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1", "2", "3", "4", "5","6", "7", "8", "9", "10", "11", "12", "13") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 0, None): # skip header
            yield line[0], line[1].encode('utf8'),None # label, text_a, text_b


def dataset_class(task):
    """ Mapping from task string to Dataset Class """
    table = {'imdb': IM, 'agnews': AG, 'yahoo': YA, 'DBpedia':DB}
    return table[task]


class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError

class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) \
                   if text_b else []

        return (label, tokens_a, tokens_b)

    
class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, label_id)

class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)
        self.init_classifier_param()

    def init_classifier_param(self):
        self.classifier.weight.data.normal_(std=0.02)
        self.classifier.bias.data.fill_(0)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0])) 
        logits = self.classifier(self.drop(pooled_h))
        return logits
    

def main(task='mrpc',
         train_cfg='./model/config/train_mrpc.json',
         model_cfg='./model/config/bert_base.json',
         data_train_file='total_data/imdbtrain.tsv',
         data_test_file='total_data/IMDB_test.tsv',
         model_file=None,
         pretrain_file='./model/uncased_L-12_H-768_A-12/bert_model.ckpt',
         data_parallel=False,
         vocab='./model/uncased_L-12_H-768_A-12/vocab.txt',
         dataName='IMDB',
         max_len=300,
         mode='train'):
   

    if mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss
        
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)

            return label_id, logits
        
        
        def pseudo_labeling(mode, model, batch, global_step,ls):
            if(global_step== 0):
                predict.clear()
            input_ids, segment_ids, input_mask, label_id = batch 
            logits = model(input_ids, segment_ids, input_mask)
            logits2=F.softmax(logits)
            y_pred11, y_pred = logits2.max(1)
 
            if(mode == 'finetuning_pseudo_labeling'):
                for i in range(0, len(input_ids)):
                    predict.append([y_pred11[i].item(), y_pred[i].item() ,data0[global_step*cfg.batch_size+i]])
            elif(mode == 'superficial_pseudo_labeling'):
                for i in range(0, len(input_ids)):
                    predict.append([y_pred11[i].item(), y_pred[i].item() ,data0[global_step*cfg.batch_size+i]])
                
            if(global_step==ls-1):
                predictS = sorted(predict, reverse = True)

                
                if(mode == 'finetuning_pseudo_labeling'):
                    rla = {}
                    for line in predictS:
                        if(line[0] > 0.9 and str(line[1]) not in rla):
                            rla[str(line[1])] = 1
                        elif(line[0] > 0.9):
                            rla[str(line[1])] += 1
                    minNum = 987654321
                    labelN = 0
                    for key, value in rla.items():
                        labelN += 1
                        minNum = min(minNum, value)
                    if(labelN != labelNum or minNum == 1):
                        minNum = 0
                    print("minNum##############: ", minNum) 
                    f_labeled = open(data_labeled_file, 'a', encoding='utf-8', newline='')
                    w_labeled = csv.writer(f_labeled, delimiter='\t')
                    
                    f_remain = open("remain_data.tsv", 'w', encoding='utf-8', newline='')
                    w_remain = csv.writer(f_remain, delimiter='\t')
                    rla = {}
                    for line in predictS:
                        if(str(line[1]) not in rla):
                            rla[str(line[1])] = 1
                        elif(rla[str(line[1])] <= minNum):
                            rla[str(line[1])] += 1
                            w_labeled.writerow([str(line[1]),line[2]])
                        else:
                            w_remain.writerow([str(line[1]),line[2]])
                    f_labeled.close()
                    f_remain.close()
                    
                    data0.clear()
                    with open("remain_data.tsv", "r", encoding='utf-8') as f:
                        lines = csv.reader(f, delimiter='\t')
                        for line in lines:
                            data0.append(line[1])

                    dataset_remain = TaskDataset("remain_data.tsv", pipeline)
                    data_iter_remain = DataLoader(dataset_remain, batch_size=cfg.batch_size, shuffle=False)

                    dataset_labeled = TaskDataset(data_labeled_file, pipeline)
                    data_iter_labeled = DataLoader(dataset_labeled, batch_size=cfg.batch_size, shuffle=True)
                    
                    data_iter_pseudo = 1

                elif(mode == 'superficial_pseudo_labeling'):
                    rla = {}
                    for line in predictS:
                        if(line[0] > 0.9 and str(line[1]) not in rla):
                            rla[str(line[1])] = 1
                        elif(line[0] > 0.9):
                            rla[str(line[1])] += 1
                    minNum = 987654321
                    labelN = 0
                    for key, value in rla.items():
                        labelN += 1
                        minNum = min(minNum, value)
                    if(labelN != labelNum or minNum == 1):
                        minNum = 0
                    
                    dev_labeled_data = []
                    with open(data_labeled_file, "r", encoding='utf-8') as f:
                        lines = csv.reader(f, delimiter='\t')
                        for line in lines:
                            dev_labeled_data.append([line[0], line[1]])

                    with open(data_dev_file, "r", encoding='utf-8') as f:
                        lines = csv.reader(f, delimiter='\t')
                        for line in lines:
                            dev_labeled_data.append([line[0], line[1]])
                    
                    tempDict = copy.deepcopy(Dict2)
                    for line in dev_labeled_data:
                        lineT = line[1].split(' ')
                        flagNum = 0
                        for lineTT in lineT:
                            if(lineTT.lower() in words2):
                                if(lineTT.lower() not in tempDict[line[0]]):
                                    tempDict[line[0]][lineTT.lower()] = 1
                                else:
                                    tempDict[line[0]][lineTT.lower()] += 1
                    
                    tempDList = []
                    minN = 987654321
                    for l in range(0, labelNum):
                        temp = []
                        tempD = sorted(tempDict[str(l)].items(),key=lambda x: x[1], reverse=True)
                        for key, value in tempD:
                            if(value < 2):
                                continue
                            temp.append(key)
                        minN = min(minN, len(temp))
                        tempDList.append(temp)
                    tempDict = copy.deepcopy(Dict2)

                    for l in range(0, labelNum):
                        tempDict2 = tempDList[l][:minN]
                        for ll in tempDict2:
                            if(ll not in tempDict[str(l)]):
                                tempDict[str(l)][ll] = 1
                            else:
                                tempDict[str(l)][ll] += 1 
                        
                    rla = {}
                    for line in predictS:
                        if(str(line[1]) not in rla):
                            rla[str(line[1])] = 1
                        else:
                            rla[str(line[1])] += 1
                    
                    minNum = 987654321
                    labelN = 0 
                    for key, value in rla.items():
                        labelN += 1
                        minNum = min(minNum, value)
                    if(labelN != labelNum or minNum == 1):
                        minNum = 1000
                    print("#####minNum#######: ", minNum) 

                    f_pseudo = open("pseudo_labeled_data2.tsv", 'w', encoding='utf-8', newline='')
                    w_pseudo = csv.writer(f_pseudo, delimiter='\t')

                    rla = {}
                    rla2 = {}

                    tempDD = []
                    for line in predictS:
                        if(str(line[1]) not in rla):
                            rla[str(line[1])] = 1
                            w_pseudo.writerow([str(line[1]),line[2]])
                        elif(rla[str(line[1])] < minNum):
                            rla[str(line[1])] += 1
                            w_pseudo.writerow([str(line[1]),line[2]])
                        else:
                            lineT = line[2].split(' ')
                            matchingNum = {}
                            tempDict3 = copy.deepcopy(Dict2)
                            for k in range(0, labelNum):
                                for lineTT in lineT:
                                    if(lineTT.lower() in tempDict[str(k)]):
                                        if(lineTT.lower() not in tempDict3[str(k)]):
                                            tempDict3[str(k)][lineTT.lower()] = 1
                                        else:
                                            continue
                                        
                                        if(str(k) not in matchingNum):
                                            matchingNum[str(k)] = 1
                                        else:
                                            matchingNum[str(k)] += 1
                            maxGG = 0
                            for key, value in matchingNum.items():
                                maxGG = max(maxGG, int(value))
                            label = -1
                            num =0
                            overlap = 0
                            for key, value in matchingNum.items():
                                if(maxGG == value):
                                    label = key
                                    num = value
                                    overlap +=1
                            if(label == -1 or overlap >=2 and num <=1):
                                continue
                            w_pseudo.writerow([label,line[2]])
                    
                    f_pseudo.close()

                    data_iter_remain = 1
                    data_iter_labeled = 1
                    dataset_pseudo = TaskDataset("pseudo_labeled_data2.tsv", pipeline)
                    data_iter_pseudo = DataLoader(dataset_pseudo, batch_size=cfg.batch_size, shuffle=True)

            if(global_step!=ls-1):
                data_iter_remain = 1
                data_iter_labeled = 1
                data_iter_pseudo = 1

            return data_iter_remain, data_iter_labeled, data_iter_pseudo
        
        if(dataName == "IMDB"):
            labelNum = 2
            dataName= "IMDB"
            tdataName = "imdbtrain"
            testName = "IMDB_test"
            Dict2 = {
		    "0" : {},
		    "1" : {}
		    }
        elif(dataName == "AG"):
            labelNum = 4
            dataName = "AG"
            tdataName = "agtrain"
            testName = "ag_test"
            Dict2 = {
		    "0" : {},
		    "1" : {},
		    "2" : {},
		    "3" : {}
		    }
        elif(dataName == "yahoo"):
            labelNum = 10
            dataName = "yahoo"
            tdataName = "yahootrain"
            testName = "yahoo_test"
            Dict2 = {
		    "0" : {},
		    "1" : {},
		    "2" : {},
		    "3" : {},
		    "4" : {},
		    "5" : {},
		    "6" : {},
		    "7" : {},
		    "8" : {},
		    "9" : {}
		    }
        
        
        elif(dataName == "dbpedia"):
            labelNum = 14
            dataName == "dbpedia"
            tdataName = "dbtrain"
            testName = "db_test"
            Dict2 = {
		    "0" : {},
		    "1" : {},
		    "2" : {},
		    "3" : {},
		    "4" : {},
		    "5" : {},
		    "6" : {},
		    "7" : {},
		    "8" : {},
		    "9" : {},
		    "10" : {},
		    "11" : {},
		    "12" : {},
		    "13" : {}
		    }
     
        def generating_words(res):
            stopwords=get_stop_words("./model_BERT/stopwords.txt")
            cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
            corpus = []
            for line in res:
                lineT = pre_process(line[1])
                corpus.append(lineT)
            
            word_count_vector=cv.fit_transform(corpus)
            tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
            tfidf_transformer.fit(word_count_vector)
            
            feature_names=cv.get_feature_names() 
            keywordD= {}
            for line in corpus:
                tf_idf_vector=tfidf_transformer.transform(cv.transform([line]))
                sorted_items=sort_coo(tf_idf_vector.tocoo())
                keywords=extract_topn_from_vector(feature_names,sorted_items,3)
                for k in keywords:
                    if(k not in keywordD):
                        keywordD[k] = 1	
                    else:
                        keywordD[k] += 1

            tempD = sorted(keywordD.items(),key=lambda x: x[1], reverse=True)
            
            words2 = {}
            for key, value in tempD:
                token_ab = key
                if(token_ab == '' or token_ab == ' ' or token_ab ==',' or token_ab=='.' or token_ab == 'from' or token_ab == 'are' or token_ab == 'is' or token_ab == 'and' or token_ab == 'with' or token_ab == 'may' or token_ab == 'would' or token_ab == 'could' or token_ab == 'have' or token_ab == 'has' or token_ab == 'had' or token_ab == 'was' or token_ab == 'were' or token_ab == 'this' or token_ab == 'who' or token_ab == 'that' or token_ab == 'www' or token_ab == 'http' or token_ab == 'com' or token_ab == 'those' or token_ab == 'your' or token_ab == 'not' or token_ab == 'seem' or token_ab == 'too' or token_ab == 'lol'or token_ab == 'but' or token_ab == 'these' or token_ab == 'their' or token_ab == 'can' or token_ab == 'there' or token_ab == 'gave' or token_ab == 'his'  or token_ab == 'etc' or token_ab == 'thats' or token_ab == 'though' or token_ab == 'off' or token_ab == 'she' or token_ab == 'them' or token_ab == 'huh' or token_ab == 'why' or token_ab == 'wont' or token_ab == 'any' or token_ab == 'some' or token_ab == 'its' or token_ab == 'yeah' or token_ab == 'yes' or token_ab == 'you' or token_ab == 'should' or token_ab == 'dont' or token_ab == 'anybody' or token_ab == 'than' or token_ab == 'where' or token_ab == 'for' or token_ab == 'more' or token_ab == 'will' or token_ab == 'him' or token_ab == 'its' or token_ab == 'your' or token_ab == 'wii' or token_ab == 'having' or token_ab == 'just' or token_ab == 'help'  or token_ab == 'helps' or token_ab == 'all' or token_ab == 'they' or token_ab == 'take' or token_ab == 'the' or token_ab == 'what' or token_ab == 'need' or token_ab == 'make' or token_ab == 'about' or token_ab == 'then' or token_ab == 'when' or token_ab == 'does' or token_ab == 'ask'  or token_ab == 'much' or token_ab == 'man' or token_ab == 'know' or token_ab == 'how' or token_ab == 'look' or token_ab == 'like' or token_ab == 'one' or token_ab == 'think' or token_ab == 'tell' or token_ab == 'find' or token_ab == 'cant' or token_ab == 'now' or token_ab == 'try' or token_ab == 'give' or token_ab == 'answer' or token_ab == 'her' or token_ab == 'out' or token_ab == 'get' or token_ab == 'because'  or token_ab == 'myself' or token_ab == 'wants' or token_ab == 'movie' or token_ab == 'film' or token_ab == 'films') : 
                    continue
                if(len(key) <=2):
                    continue
                if(key not in words2):
                    words2[key] = 1
                else:
                    words2[key] = 1
            return words2
        

        curNum=1
        cfg = train.Config.from_json(train_cfg)
        model_cfg = models.Config.from_json(model_cfg)
        set_seeds(cfg.seed)


        for kkk in  range(0,5):
            tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
            TaskDataset = dataset_class(task) # task dataset class according to the task
            pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                        AddSpecialTokensWithTruncation(max_len),
                        TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, max_len)]



            ################# data reading ################
            data_unlabeled_file = "./data/"+dataName + "_unlabeled" + str(kkk+1)+".tsv"
            data_dev_file = "./data/" + dataName + "_dev" + str(kkk+1)+".tsv"
            data_labeled_file = "./data/" + dataName + "_labeled" + str(kkk+1)+".tsv"
            data_total_file = "./total_data/" + tdataName + ".tsv"
            data_test_file = "./total_data/" + testName + ".tsv"


            f_total = open(data_total_file, 'r', encoding='utf-8')
            r_total = csv.reader(f_total, delimiter='\t')

            allD=[]
            for line in r_total:
                allD.append([line[0],line[1]])
            f_total.close()

            for ii in range(0, kkk+1):
                random.shuffle(allD)
            
            #num_data = 5010* labelNum
            num_data_dev_temp = 10 * labelNum
            num_data_dev = 5 * labelNum
            num_data_labeled = 5 * labelNum
            num_data_unlabeled = 200000 - num_data_dev_temp
           # num_data_unlabeled = len(allD) - num_data_dev_temp
            
            print("num_data_dev#: ", num_data_dev)
            print("num_data_labeled#: ",num_data_labeled)
            print("num_data_unlabeled#: ",num_data_unlabeled)



            f_unlabeled = open(data_unlabeled_file, 'w', encoding='utf-8', newline='')
            w_unlabeled = csv.writer(f_unlabeled, delimiter='\t')
           
            f_dev = open(data_dev_file, 'w', encoding='utf-8', newline='')
            w_dev = csv.writer(f_dev, delimiter='\t')

            f_labeled = open(data_labeled_file, 'w', encoding='utf-8', newline='')
            w_labeled = csv.writer(f_labeled, delimiter='\t')
            allD2=[]
            tempD={}
            for line in allD:
                if(line[0] not in tempD):
                    allD2.append([line[0],line[1]])
                    tempD[line[0]] = 1
                elif(tempD[line[0]] <= int(num_data_dev_temp/labelNum)):
                    allD2.append([line[0],line[1]])
                    tempD[line[0]] += 1
                elif(tempD[line[0]] <= int(num_data_dev_temp/labelNum)+int(num_data_unlabeled/labelNum)):
                    allD2.append([line[0],line[1]])
                    tempD[line[0]] += 1

            words2 = generating_words(allD2) # 사용하는 데이터만큼 가져와서 tf-idf를 통해 핵심단어 사전을 구축. words2는 핵심단어 사전을 가진 dictionary임
            
            tempD={}
            tempD2={}
            tempD3={}
            for k in range(0, labelNum):
                tempD[str(k)] = 0
                tempD2[str(k)] = 0
                tempD3[str(k)] = 0
            
            if('IMDB' in dataName):
                max_len2 = 250
                min_len = 150
            else:
                max_len2 = 250
                min_len = 0        

            dev_num =0
            labeled_num =0
            unlabeled_num = 0
            for line in allD2:
                lineWords = len(line[1].split(" "))
                if(tempD[line[0]] < int(num_data_dev/labelNum)):
                    w_dev.writerow([line[0],line[1]])
                    dev_num+=1
                    tempD[line[0]] += 1
                elif(tempD2[line[0]] < int(num_data_labeled/labelNum) and lineWords >= min_len and lineWords <=max_len2):
                    labeled_num+=1
                    w_labeled.writerow([line[0],line[1]])
                    tempD2[line[0]] += 1
                elif(tempD3[line[0]] <= int(num_data_unlabeled/labelNum)):
                    unlabeled_num+=1
                    w_unlabeled.writerow([line[0],line[1]])
                    tempD3[line[0]] += 1
            print('number of dev data :' , dev_num)
            print('number of labeled data :' , labeled_num)
            print('number of unlabeled data :' , unlabeled_num)

            f_unlabeled.close()                
            f_labeled.close()                
            f_dev.close()                


            
            ################# data reading done!!!!!!! ################

            dev_labeled_data = []
            tempDict = copy.deepcopy(Dict2) 
            with open(data_labeled_file, "r", encoding='utf-8') as f:
                lines = csv.reader(f, delimiter='\t')
                for line in lines:
                    dev_labeled_data.append([line[0], line[1]])
           
            with open(data_dev_file, "r", encoding='utf-8') as f:
                lines = csv.reader(f, delimiter='\t')
                for line in lines:
                    dev_labeled_data.append([line[0], line[1]])
            


            tempDict = copy.deepcopy(Dict2) 
            for line in dev_labeled_data:
                lineT = line[1].split(' ')
                flagNum = 0
                for lineTT in lineT:
                    if(lineTT.lower() in words2):
                        if(lineTT.lower() not in tempDict[line[0]]):
                            tempDict[line[0]][lineTT.lower()] = 1
                        else:
                            tempDict[line[0]][lineTT.lower()] += 1
            
            tempDict333 = tempDict.copy()
            tempDList = []
            minN = 987654321
            for l in range(0, labelNum):
                temp = []
                tempD = sorted(tempDict[str(l)].items(),key=lambda x: x[1], reverse=True)
                for key, value in tempD:
                    temp.append(key)
                minN = min(minN, len(temp))
                print('len(temp):', len(temp))
                tempDList.append(temp)

            tempDict = copy.deepcopy(Dict2)
            for l in range(0, labelNum):
                tempDict2 = tempDList[l][:minN]
                for ll in tempDict2:
                    if(ll not in tempDict[str(l)]):
                        tempDict[str(l)][ll] = 1
                    else:
                        tempDict[str(l)][ll] += 1 

            f_pseudo = open("pseudo_labeled_data.tsv", 'w', encoding='utf-8', newline='')
            w_pseudo = csv.writer(f_pseudo, delimiter='\t')

            rla = {}
            with open(data_unlabeled_file, "r", encoding='utf-8') as f:
                lines = csv.reader(f, delimiter='\t')
                for line in lines:
                    lineT = line[1].split(' ')
                    matchingNum = {}
                    tempDict3 = copy.deepcopy(Dict2)
                    for k in range(0, labelNum):
                        for lineTT in lineT:
                            if(lineTT.lower() in tempDict[str(k)]):
                                if(lineTT.lower() not in tempDict3[str(k)]):
                                    tempDict3[str(k)][lineTT.lower()] = 1
                                else:
                                    continue
                                
                                if(str(k) not in matchingNum):
                                    matchingNum[str(k)] = 1
                                else:
                                    matchingNum[str(k)] += 1
                    maxN = 0
                    for key, value in matchingNum.items():
                        maxN = max(maxN, int(value))
                    label = -1
                    num =0
                    overlap = 0
                    for key, value in matchingNum.items():
                        if(maxN == value):
                            label = key
                            num = value
                            overlap +=1
                    if(label == -1 or overlap >=2 or num<=1):
                        continue
                    w_pseudo.writerow([label,line[1]])

            f_pseudo.close()
            print("----------------temp data generating done!!!!!!!!------------------")


            dataset_unlabeled = TaskDataset(data_unlabeled_file, pipeline)
            data_iter_unlabeled = DataLoader(dataset_unlabeled, batch_size=cfg.batch_size, shuffle=False)

            dataset_dev = TaskDataset(data_dev_file, pipeline)
            data_iter_dev = DataLoader(dataset_dev, batch_size=cfg.batch_size, shuffle=True)
            
            dataset_labeled = TaskDataset(data_labeled_file, pipeline)
            data_iter_labeled = DataLoader(dataset_labeled, batch_size=cfg.batch_size, shuffle=True)
            
            dataset_test = TaskDataset(data_test_file, pipeline)
            data_iter_test = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=False)

            dataset_twt = TaskDataset("pseudo_labeled_data.tsv", pipeline)
            data_iter_twt = DataLoader(dataset_twt, batch_size=cfg.batch_size, shuffle=True)

            curNum+=1
            criterion = nn.CrossEntropyLoss()
            model = Classifier(model_cfg, labelNum)

            trainer = train.Trainer(cfg,
                                    dataName,
                                    model,
                                    data_iter_unlabeled,
                                    data_iter_dev,
                                    data_iter_labeled,
                                    data_iter_test,
                                    data_iter_twt,
                                    get_device(),kkk+1)



            predict=[]


            data0=[]
            data1=[]

            with open(data_unlabeled_file, "r", encoding='utf-8') as f:
                lines = csv.reader(f, delimiter='\t')

                for line in lines:
                    data0.append(line[1])
                    data1.append(line[1])
          
            trainer.train(model_file, pretrain_file, get_loss,evaluate,pseudo_labeling,data_parallel)


if __name__ == '__main__':
    fire.Fire(main)
