import dgl
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, average_precision_score
from tqdm import tqdm
from utils import *
import time
from memory_profiler import profile
import random


class Evaluator():
    def __init__(self, params, model, data):
        self.params = params
        self.model = model
        self.data = data
        self.loss_pred = nn.BCELoss()
        self.cl_loss = Multi_View_CL_Loss()
        
    def global_ranking(self, true_labels, pred_scores, top=1, typ='recall'):
        true_labels, pred_scores = np.array(true_labels), np.array(pred_scores)
        sorted_index = np.argsort(-pred_scores)
        #top_num = int(top * len(true_labels))
        top_num = top
        sorted_true_labels = true_labels[sorted_index[:top_num]]
        if typ == 'precision' :
            value = float(sorted_true_labels.sum())/float(top_num)
        elif typ == 'recall' :
            all_positive = true_labels.sum()
            value = float(sorted_true_labels.sum())/float(all_positive)
        return value
    
    @profile(precision=4, stream=open('mem.log','w+')) #指标换成对所有数据的评估
    def eval(self):
        dataloader = DataLoader( self.data, batch_size=self.params.batch_size*2, shuffle=True, num_workers=4, collate_fn=collate_dgl)
        bar = tqdm(enumerate(dataloader))
        self.model.eval()
        torch.cuda.empty_cache()
        auc_all = list(); aupr_all = list(); f1_all = list()
        loss_all ={'pred':list(), 'cl':list(), 'l2':list()}
        
        label_all = list(); pred_all = list()
        
        with torch.no_grad():
            for b_idx, batch in bar:
                gene_subgraphs, gene1_embs, gene2_embs, sl_lbs, gene1s, gene2s = move_batch_to_device(*batch, device=self.params.device)
                if gene1_embs.size()[0] == 1 :  # some batches only have one data
                    continue
                graph_rep, seq_rep, prediction_score, prediction_emb = self.model(gene_subgraphs, gene1_embs, gene2_embs, gene1s, gene2s)
                try:
                    pred = torch.squeeze(prediction_score).clone().detach().cpu().numpy().tolist()
                    label = sl_lbs.clone().detach().cpu().numpy().tolist()
                    auc = metrics.roc_auc_score(label, pred)
                    p, r, t = metrics.precision_recall_curve(label, pred)
                    aupr = metrics.auc(r, p)
                    pred_scores = [1 if i else 0 for i in (np.array(pred) >= 0.5)]
                    f1 = metrics.f1_score(label, pred_scores)
                    bar.set_description(f'Evaluate iteration:{str(b_idx+1)}/{len(dataloader)}' )
                    torch.cuda.empty_cache()
                    
                    auc_all.append(auc)
                    aupr_all.append(aupr)
                    f1_all.append(f1)
                except:
                    #print("unbalanced data in batch")
                    pass
                finally:
                    label_all += label
                    pred_all += pred
                #loss_all['pred'].append(loss_pred)
                #loss_all['cl'].append(loss_cl) 
                #loss_all['l2'].append(l2_loss)     
                
                
            pred_scores_all = [1 if i else 0 for i in (np.array(pred_all) >= 0.5)]
            auc = metrics.roc_auc_score(label_all, pred_all)
            p, r, t = metrics.precision_recall_curve(label_all, pred_all)
            aupr = metrics.auc(r, p)
            f1 = metrics.f1_score(label_all, pred_scores_all)
            acc = metrics.accuracy_score(label_all, pred_scores_all)
            precision = metrics.precision_score(label_all, pred_scores_all)
            recall = metrics.recall_score(label_all, pred_scores_all)
            
            global_pre3 = self.global_ranking(label_all, pred_all, top=3, typ='precision')
            global_pre5 = self.global_ranking(label_all, pred_all, top=5, typ='precision')
            global_pre10 = self.global_ranking(label_all, pred_all, top=10, typ='precision')
            global_pre20 = self.global_ranking(label_all, pred_all, top=20, typ='precision')
            
            global_recall3 = self.global_ranking(label_all, pred_all, top=3, typ = 'recall' )
            global_recall5 = self.global_ranking(label_all, pred_all, top=5, typ = 'recall' )
            global_recall10 = self.global_ranking(label_all, pred_all, top=10, typ = 'recall' )
            global_recall20 = self.global_ranking(label_all, pred_all, top=20, typ = 'recall' )
    
        
            #random result
            random_pred = [random.random() for i in label_all]
            random_pred_scores_all  = [1 if i else 0 for i in (np.array(random_pred) >= 0.5)]
            random_auc = metrics.roc_auc_score(label_all, random_pred)
            random_p, random_r, t = metrics.precision_recall_curve(label_all, random_pred)
            random_aupr = metrics.auc(random_r, random_p)
            random_f1 = metrics.f1_score(label_all, random_pred_scores_all)
            random_acc = metrics.accuracy_score(label_all, random_pred_scores_all)
            random_precision = metrics.precision_score(label_all, random_pred_scores_all)
            random_recall = metrics.recall_score(label_all, random_pred_scores_all)
            
            
            random_global_pre3 = self.global_ranking(label_all, random_pred, top=3, typ='precision')
            random_global_pre5 = self.global_ranking(label_all, random_pred, top=5, typ='precision')
            random_global_pre10 = self.global_ranking(label_all, random_pred, top=10, typ='precision')
            random_global_pre20 = self.global_ranking(label_all, random_pred, top=20, typ='precision')
            
            random_global_recall3 = self.global_ranking(label_all, random_pred, top=3, typ = 'recall' )
            random_global_recall5 = self.global_ranking(label_all, random_pred, top=5, typ = 'recall' )
            random_global_recall10 = self.global_ranking(label_all, random_pred, top=10, typ = 'recall' )
            random_global_recall20 = self.global_ranking(label_all, random_pred, top=20, typ = 'recall' )
            
            
             
            

        return { 'aupr_batch_mean':np.mean(aupr_all), 'auc_batch_mean':np.mean(auc_all), 'f1_batch_mean':np.mean(f1_all), 'aupr':aupr, 'auc':auc, 'f1':f1, 'precision':precision, 'recall':recall, 'accuracy':acc,'GlobalPrecision@3':global_pre3,'GlobalPrecision@5':global_pre5,'GlobalPrecision@10':global_pre10,'GlobalPrecision@20':global_pre20, 'GlobalRecall@3':global_recall3,'GlobalRecall@5':global_recall5,'GlobalRecall@10':global_recall10,'GlobalRecall@20':global_recall20, 'random_aupr':random_aupr, 'random_auc':random_auc, 'random_f1':random_f1, 'random_precision':random_precision, 'random_recall':random_recall, 'random_accuracy':random_acc, 'random_GlobalPrecision@3':random_global_pre3,'random_GlobalPrecision@5':random_global_pre5, 'random_GlobalPrecision@10':random_global_pre10,'random_GlobalPrecision@20':random_global_pre20, 'random_GlobalRecall@3':random_global_recall3,'random_GlobalRecall@5':random_global_recall5,'random_GlobalRecall@10':random_global_recall10,'random_GlobalRecall@20':random_global_recall20,  }
                
                
                 
            
        
        
        
        
        

        
        
    