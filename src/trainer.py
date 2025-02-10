import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.nn as nn
import torch.optim as optim
from utils import *
from sklearn import metrics
from evaluator import Evaluator
import time
from memory_profiler import profile
import logging
import os
import traceback
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import StepLR



#trainer
class Trainer():
    def __init__(self, params, model, train_data, valid_evaluator, test_evaluator, nr_rate, result_suffix=""):
        self.model = model.to(params.device)
        
        model_params = list(self.model.parameters())
        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=params.l2)
        self.scheduler = StepLR(self.optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma) 
        
        self.train_data = train_data
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator
        self.params = params
        self.loss_pred = nn.BCELoss()
        self.cl_loss = Multi_View_CL_Loss()
        self.best_metric = 0
        self.best_test_result = dict()
        self.nr_rate = nr_rate
        
        self.result_suffix = result_suffix
        
    
    def reset_trainer_state(self):
        self.best_metric = 0
        self.best_test_result = dict()
        self.best_epoch = None
        self.wo_improvement = 0 
        
        
    @profile(precision=4, stream=open('mem.log','w+'))  
    def train_epoch(self, epoch):
        dataloader = DataLoader( self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=4, collate_fn=collate_dgl)
        bar = tqdm(enumerate(dataloader))
        self.model.train()
        torch.cuda.empty_cache()
        train_pred_all = list(); train_label_all = list()
        train_auc_all = list(); train_aupr_all = list(); train_f1_all = list()
        train_loss_all ={'pred':list(), 'cl_level_1':list(),'cl_level_2':list(), 'l2':list(), 'all':list()}
        best_epoch = False

        for b_idx, batch in bar:
            gene_subgraphs, gene1_embs, gene2_embs, sl_lbs, gene1s, gene2s = move_batch_to_device(*batch, device=self.params.device)
            if gene1_embs.size()[0] == 1 :  # some batches only have one data
                continue
            
            self.optimizer.zero_grad()
            graph_rep, seq_rep, prediction_score, prediction_emb = self.model(gene_subgraphs, gene1_embs, gene2_embs, gene1s, gene2s)
            loss_pred = self.loss_pred( torch.squeeze(prediction_score), sl_lbs )
            
            loss_cl_level_1 = 0.0; loss_cl_level_2 = 0.0
            try:
                loss_cl_level_2, loss_cl_level_1 = self.cl_loss( graph_rep, seq_rep, sl_lbs, typ="sum", sample_num=self.params.loss_cl_sample, prediction_emb=prediction_emb ) #2before1
                if self.params.wo_cl_loss_level_1 : 
                    loss_cl_level_1 = 0.0
                if self.params.wo_cl_loss_level_2 :
                    loss_cl_level_2 = 0.0
            except:
                if self.params.wo_cl_loss_level_1 and self.params.wo_cl_loss_level_2:
                    pass
                else :
                    print("None")
                    continue
            
            l2_loss = []
            for module in self.model.modules():
                if type(module) is nn.Linear:
                    l2_loss.append((module.weight ** 2).sum() / 2)
            l2_loss = torch.sum(torch.tensor(l2_loss, requires_grad=True))
            loss = loss_pred + self.params.loss_cl_level_2*loss_cl_level_2 + self.params.loss_cl_level_1*loss_cl_level_1 + self.params.l2*l2_loss 
            loss.backward() 
            train_loss_all['l2'].append(l2_loss.detach().cpu().numpy())
            train_loss_all['all'].append(loss.detach().cpu().numpy())
            #print(loss_pred.requires_grad, loss_cl.requires_grad, l2_loss.requires_grad)
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()
                
            bar.set_description(f'Trian iteration:{str(b_idx+1)}/{len(dataloader)} loss_train:{loss.cpu().detach().numpy()}' )
            
            with torch.no_grad(): 
                try :
                    # calculate train metric
                    train_pred = torch.squeeze(prediction_score).clone().detach().cpu().numpy().tolist()
                    train_label = sl_lbs.clone().detach().cpu().numpy().tolist()
                    train_auc = metrics.roc_auc_score(train_label, train_pred)
                    p, r, t = metrics.precision_recall_curve(train_label, train_pred)
                    train_aupr = metrics.auc(r, p)
                    pred_scores = [1 if i else 0 for i in (np.array(train_pred) >= 0.5)]
                    train_f1 = metrics.f1_score(train_label, pred_scores)
            
                    train_auc_all.append(train_auc)
                    train_aupr_all.append(train_aupr)
                    train_f1_all.append(train_f1)
                    train_loss_all['pred'].append(loss_pred.detach().cpu().numpy())
                    try:
                        train_loss_all['cl_level_1'].append(loss_cl_level_1.detach().cpu().numpy())
                    except:
                        train_loss_all['cl_level_1'].append(loss_cl_level_1)      
                    try:
                        train_loss_all['cl_level_2'].append(loss_cl_level_2.detach().cpu().numpy()) 
                    except:
                        train_loss_all['cl_level_2'].append(loss_cl_level_2)   
                except :
                    continue
                
            torch.cuda.empty_cache() 
            #time.sleep(0.003)
        
        
        valid_result = self.validation()
        logging.info(f'[ Validation Performance: {valid_result} in {time.time()} ]')
        if epoch % 5 == 0 :
            test_result = self.test()
            logging.info(f'[ Test Performance: {test_result} in {time.time()} ]')
            
            
        if valid_result[self.params.model_criterion] >= self.best_metric :
            self.wo_improvement = 0
            if epoch % 5 != 0 : 
                test_result = self.test()
                
            if self.params.store_model :
                torch.save(self.model.state_dict(), f'{self.params.result_dir}/{self.params.expdir}/{self.params.cell_line}/best_model_{self.result_suffix}.pth')      #              
            np.save(f'{self.params.result_dir}/{self.params.expdir}/{self.params.cell_line}/metrics_{self.result_suffix}.npy', test_result, allow_pickle=True)
            
            self.best_test_result = test_result
            best_epoch = True
            logging.info(f' *****Better model found w.r.t {self.params.model_criterion}. Saved it!******')
            logging.info(f'[ Test Performance of better model: {test_result} in {time.time()} ]\n')
        else:
            self.wo_improvement += 1 
            
            
        torch.cuda.empty_cache()
        train_loss_all['pred'] = np.nanmean(train_loss_all['pred'])
        train_loss_all['cl_level_1'] = np.nanmean(train_loss_all['cl_level_1'])
        train_loss_all['cl_level_2'] = np.nanmean(train_loss_all['cl_level_2'])
        train_loss_all['l2'] = np.nanmean(train_loss_all['l2'])
        train_loss_all['all'] = np.nanmean(train_loss_all['all'])
        
        
        return  { 'train_aupr':np.nanmean(train_aupr_all), 'train_auc':np.nanmean(train_auc_all), 'train_f1':np.nanmean(train_f1_all), 'loss':train_loss_all, 'valid_aupr':valid_result['aupr'], 'valid_auc':valid_result['auc'], 'valid_f1':valid_result['f1'], 'test_aupr':test_result['aupr'], 'test_auc':test_result['auc'], 'test_f1':test_result['f1'],  'test_accuracy':test_result['accuracy'], 'test_precision':test_result['precision'], 'test_recall':test_result['recall'], 'test_random_aupr':test_result['random_aupr'], 'test_random_auc':test_result['random_auc'], 'test_random_f1':test_result['random_f1'],  'test_random_accuracy':test_result['random_accuracy'], 'test_random_precision':test_result['random_precision'], 'test_random_recall':test_result['random_recall'] }, best_epoch
         
        
    def validation(self):
        #valid_evaluator = Evaluator(self.params, self.model, self.valid_data )
        result_dict = self.valid_evaluator.eval()
        return result_dict
    
    
    def test(self):
        result_dict = self.test_evaluator.eval()
        return result_dict


    def train(self):
        self.reset_trainer_state()
        for epoch in range(1, self.params.num_epochs + 1):
            if self.wo_improvement <= 4 :
                time_start = time.time()
                result_dict, best_epoch = self.train_epoch(epoch=epoch)
                loss, auc, aupr, f1= result_dict['loss'], result_dict['train_auc'], result_dict['train_aupr'], result_dict['train_f1']
                logging.info( f'Epoch {epoch} with loss: {loss}, train_aupr: {aupr}, train_auc: {auc}, train_f1: {f1} in {time.time()-time_start}' )
                if best_epoch:
                    self.best_epoch = epoch
            else :
                logging.info(f'[ Training is ended before epoch {epoch} ]')
                break
        logging.info(f'[ Test Performance of the best model: {self.best_test_result} in epoch {self.best_epoch} ]')
        
            
            
            
        
        
                
            

    
            
