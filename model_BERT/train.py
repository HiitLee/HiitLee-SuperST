import os
import json
from typing import NamedTuple
from tqdm import tqdm
import checkpoint
import torch
import torch.nn as nn
import optim
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score
from random import randint
from pytorchtools import EarlyStopping
from transformers import AdamW, get_linear_schedule_with_warmup 
from torch.optim import Adam
class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 1e-3 # learning rate
    n_epochs: int = 100 # the number of epoch
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, dataName,model,data_iter_unlabeled,data_iter_dev,data_iter_labeled,data_iter_test, data_iter_pseudo,device,kkk):
        self.cfg = cfg # config for training : see class Config
        self.dataName = dataName
        self.model = model
        self.data_iter_unlabeled = data_iter_unlabeled # iterator to load data
        self.data_iter_dev = data_iter_dev # iterator to load data
        self.data_iter_labeled = data_iter_labeled # iterator to load data
        self.data_iter_test = data_iter_test # iterator to load data
        self.data_iter_pseudo = data_iter_pseudo # iterator to load data
        self.device = device # device name
        self.kkk = kkk

    def superficial_learning(self, model_file, pretrain_file, model, dataset_pseudo, get_loss):
        model = torch.load(pretrain_file)

        optimizer = AdamW(model.parameters(), lr=1e-6, correct_bias = False)
        for k in range(0, 1):
            global_step = 0 # global iteration steps regardless of epochs
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(dataset_pseudo, desc='Iter (loss=X.XXX)')
            model.train()
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                loss = get_loss(model, batch, global_step).mean() # mean() for Data Parallelism
                loss.backward()
                optimizer.step()
                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
            print('Train Average Loss %5.3f'%(loss_sum/(i+1)))
        return model

    def fine_tuning(self,mode, model, dataset_labeled, dataset_dev,dataset_test,  model_save,evaluate, get_loss, result_name, num_a):
        valid_losses = []
        optimizer = AdamW(
        	model.parameters(),
        	lr = 1e-5,
                correct_bias = False
        )
        best_acc = 0
        best_loss = -1.0
        test_accs=[]
        for e in range(0, 5):
            global_step = 0 # global iteration steps regardless of epochs
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(dataset_labeled, desc='Iter (loss=X.XXX)')
            model.train()
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                loss = get_loss(model, batch, global_step).mean() # mean() for Data Parallelism
                loss.backward()
                optimizer.step()
                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
	    
            print('@@@@Train Average Loss %5.3f'%(loss_sum/(i+1)))
            model.eval()# evaluation mode
            
            val_loss, val_acc = self.validate('dev',model, dataset_dev, model_save, evaluate, get_loss, result_name, num_a, e)
            print("epoch {}, val acc {}, val_loss {}".format(e+1, val_acc, val_loss))
            
            
            if val_acc >= best_acc:
                if(val_acc == best_acc):
                    if(best_loss < val_loss):
                        continue
                best_loss = val_loss
                best_acc = val_acc
                torch.save(model.state_dict(), model_save)
                test_loss, test_acc = self.validate('test',model, dataset_test, model_save, evaluate, get_loss, result_name,num_a, e)
                test_accs.append(test_acc)
                print("epoch {}, test acc {},test loss {}".format(e+1, test_acc, test_loss))

        print('Best val_acc:')
        print(best_acc)

        ddf = open(result_name,'a', encoding='UTF8')
        print('Test acc:')
        print(test_accs)
        test_accs2=""
        for kk in test_accs:
            test_accs2 += str(kk) + ' '

        ddf.write("num_a: " + str(num_a) + " test_acc : "+str(test_accs2)+'\n')
        ddf.close()
    def validate(self, mode, model, dataset_dev, model_load,evaluate,get_loss, result_name, num_a, e):
        model.eval()# evaluation mode
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        
        global_step = 0
        with torch.no_grad():
            iter_bar = tqdm(dataset_dev, desc='Iter (f1-score=X.XXX)')
            for batch in iter_bar:
                batch = [t.to(self.device) for t in batch]
                input_ids, segment_ids, input_mask, label_id = batch
                loss = get_loss(model, batch, global_step).mean() # mean() for Data Parallelism
                targets, outputs = evaluate(model, batch) # accuracy to print
                _, predicted = torch.max(outputs.data, 1)
                correct += (np.array(predicted.cpu()) ==
                            np.array(targets.cpu())).sum()
                loss_total += loss.item() * input_ids.shape[0]
                total_sample += input_ids.shape[0]
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                
                
        acc_total = correct/total_sample
        loss_total = loss_total/total_sample
        
        return loss_total,acc_total

    def pseduo_labeling(self, mode, model, dataset, model_load, pseudo_labeling):
        model.load_state_dict(torch.load(model_load))
        global_step1 = 0
        model.eval()
        iter_bar = tqdm(dataset, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                data_iter_remain, data_iter_labeled, data_iter_pseudo = pseudo_labeling(mode, model,batch,global_step1,len(iter_bar)) 
                global_step1+=1
                result2  = 0
                iter_bar.set_description('Iter(pseduo_labeling=%5.3f)'%result2)
        return data_iter_remain, data_iter_labeled, data_iter_pseudo
 
    def train(self, model_file, pretrain_file, get_loss, evaluate, pseudo_labeling, data_parallel=False):
     
        """ Train Loop """
        self.model.train() # train mode
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
       
        print("dataName#########:", self.dataName)
        t =  self.kkk
        if(self.dataName == 'IMDB'):
            model_save_name = "./IMDB_model_save/checkpoint_bert_"+str(t)+".pt"
            model_save_name1 = "./IMDB_model_save/checkpoint_bert1_"+str(t)+".pt"
            result_name = "./result/result_IMDB.txt"
        elif(self.dataName == "AG"):
            model_save_name = "./AGNews_model_save/checkpoint_bert_"+str(t)+".pt"
            model_save_name1 = "./AGNews_model_save/checkpoint_bert1_"+str(t)+".pt"
            result_name = "./result/result_AGNews.txt"
        elif(self.dataName == "dbpedia"):
            model_save_name = "./DBpedia_model_save/checkpoint_bert_"+str(t)+".pt"
            model_save_name1 = "./DBpedia_model_save/checkpoint_bert1_"+str(t)+".pt"
            result_name = "./result/result_DBpedia.txt"
        elif(self.dataName == "yahoo"):
            model_save_name = "./yahoo_model_save/checkpoint_bert_"+str(t)+".pt"
            model_save_name1 = "./yahoo_model_save/checkpoint_bert1_"+str(t)+".pt"
            result_name = "./result/result_yahoo.txt"

        
        model = nn.DataParallel(model)         
        torch.save(model, model_save_name1)
        num_a=0
        global_step = 0 # global iteration steps regardless of epochs
        global_step3 = 0

        print("self.cfg.n_epochs#:", self.cfg.n_epochs)
        ddf = open(result_name,'a', encoding='UTF8')
        ddf.write("############################################"+str(t)+": ramdom_samplimg###########################################"+'\n')
        ddf.close()
        
        for e in range(self.cfg.n_epochs):
            if(e==0):
                model = self.superficial_learning(model_file, model_save_name1, model, self.data_iter_pseudo,get_loss)
                torch.cuda.empty_cache()
                
                self.fine_tuning("f",model, self.data_iter_labeled, self.data_iter_dev,self.data_iter_test,  model_save_name, evaluate, get_loss, result_name, num_a)
                torch.cuda.empty_cache()
                
                data_iter_remain, data_iter_labeled, _ = self.pseduo_labeling("finetuning_pseudo_labeling", model, self.data_iter_unlabeled, model_save_name, pseudo_labeling)
                _, _, data_iter_pseudo = self.pseduo_labeling("superficial_pseudo_labeling", model, data_iter_remain, model_save_name, pseudo_labeling)
                torch.cuda.empty_cache()
            else:
                model = self.superficial_learning(model_file, model_save_name1, model, data_iter_pseudo,get_loss)
                torch.cuda.empty_cache()
                
                self.fine_tuning("s",model, data_iter_labeled, self.data_iter_dev,self.data_iter_test,  model_save_name, evaluate, get_loss, result_name, num_a)
                torch.cuda.empty_cache()
                
                if(len(data_iter_remain) <=1 or len(data_iter_pseudo) <=1):
                    break
                data_iter_remain, data_iter_labeled, _ = self.pseduo_labeling("finetuning_pseudo_labeling", model, data_iter_remain, model_save_name, pseudo_labeling)
                _, _, data_iter_pseudo = self.pseduo_labeling("superficial_pseudo_labeling", model, data_iter_remain, model_save_name, pseudo_labeling)
                torch.cuda.empty_cache()

            
            num_a += 1
                
		 
    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))
        
        
        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts
                
                
    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

        

