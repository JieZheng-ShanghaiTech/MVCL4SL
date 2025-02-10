import torch 
from torch.utils.data import DataLoader
import dgl
import torch
import torch.nn as nn 
import numpy as np

def move_batch_to_device(gene_subgraphs, gene1_embs, gene2_embs, sl_lbs, gene1s=None, gene2s=None, gene_subgraph_li=None, device=None):
    gene1_embs, gene2_embs, sl_lbs = gene1_embs.to(device), gene2_embs.to(device), sl_lbs.to(device) 
    gene_subgraphs = gene_subgraphs.to(device)
    return gene_subgraphs, gene1_embs, gene2_embs, sl_lbs, gene2s, gene2s 
        
        
def collate_dgl(samples):  #move to device
    gene_subgraph, gene1_emb, gene2_emb, sl_label, gene1, gene2 = zip(*samples)
    batched_subgraph = dgl.batch(gene_subgraph)
    return batched_subgraph, torch.tensor(np.vstack(gene1_emb)), torch.tensor(np.vstack(gene2_emb)), torch.FloatTensor(np.hstack(sl_label)), gene1, gene2, gene_subgraph 



class Multi_View_CL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, graph_rep, seq_rep, sl_lbs, typ="sum", lower_bound=-150.0, sample_num=1, prediction_emb=None):
        sl_lbs = sl_lbs.detach().cpu().numpy()
        loss_level_2 = 0 ; loss_level_1 = 0
        positive_id = np.argwhere(sl_lbs==1).reshape(-1)
        negative_id = np.argwhere(sl_lbs==0).reshape(-1) 
        for i in range(sl_lbs.shape[0]):
            lb_i = sl_lbs[i]
            positive_id_c = positive_id.copy() 
            negative_id_c = negative_id.copy() 
            if lb_i == 1 :
                close_id = np.random.choice(positive_id_c,sample_num,replace=False)
                away_id = np.random.choice(negative_id,sample_num,replace=False)[close_id!=i]
                close_id = close_id[close_id!=i].reshape(away_id.shape)
            elif lb_i == 0 :
                close_id = np.random.choice(negative_id_c,sample_num,replace=False)
                away_id = np.random.choice(positive_id,sample_num,replace=False)[close_id!=i]
                close_id = close_id[close_id!=i].reshape(away_id.shape)
                
            #print(close_id, away_id)
            graph_rep_i, seq_rep_i= graph_rep[i].repeat(close_id.shape[0]), seq_rep[i].repeat(close_id.shape[0])
            prediction_emb_i = prediction_emb[i].repeat(close_id.shape[0]) 
            graph_rep_close, seq_rep_close= torch.squeeze(graph_rep[close_id]).view((-1,)), torch.squeeze(seq_rep[close_id]).view((-1,))
        
            graph_rep_away_li = list(); seq_rep_away_li = list(); prediction_emb_away_li = list() 
            for aid in away_id :
                graph_rep_away_li.append( torch.squeeze(graph_rep[aid]).repeat(close_id.shape[0]) ) 
                seq_rep_away_li.append( torch.squeeze(seq_rep[aid]).repeat(close_id.shape[0]) )  
                prediction_emb_away_li.append( torch.squeeze(prediction_emb[aid]).repeat(close_id.shape[0]) )
            #graph_rep_away, seq_rep_away= torch.squeeze(graph_rep[away_id]).view((-1,)), torch.squeeze(seq_rep[away_id]).view((-1,))  
            
            for k in range(len(graph_rep_away_li)) :
                graph_rep_away, seq_rep_away, prediction_emb_away = graph_rep_away_li[k], seq_rep_away_li[k], prediction_emb_away_li[k] 
                prediction_emb_close= torch.squeeze(prediction_emb[close_id]).view((-1,))
                loss_1 = torch.log( torch.sigmoid( torch.clip( torch.dot(graph_rep_i, seq_rep_close) - torch.dot(graph_rep_i, seq_rep_away), min=-70 ) ) ) #permutation
                loss_2 = torch.log( torch.sigmoid( torch.clip(torch.dot(seq_rep_i, graph_rep_close) - torch.dot(seq_rep_i, graph_rep_away), min=-70 ) ) )
                loss_level_1 += torch.log( torch.sigmoid( torch.clip(torch.dot(prediction_emb_i, prediction_emb_close) - torch.dot(prediction_emb_i, prediction_emb_away), min=-70 ) ) ) 
                loss_level_2 += loss_1 + loss_2
                #print(loss_level_1,loss_level_2) 
            
        if typ == "sum" :
            return loss_level_2, loss_level_1
        
        
        
#原来的写法 如果batch中某个样本只有一个 会不断抽取那个id
"""
def forward(self, graph_rep, seq_rep, sl_lbs, typ="sum", lower_bound=-150.0, sample_num=1, prediction_emb=None):
        sl_lbs = sl_lbs.detach().cpu().numpy()
        loss_level_2 = 0 ; loss_level_1 = 0
        positive_id = np.argwhere(sl_lbs==1).reshape(-1)
        negative_id = np.argwhere(sl_lbs==0).reshape(-1) 
        for i in range(sl_lbs.shape[0]):
            lb_i = sl_lbs[i]
            positive_id_c = positive_id.copy() 
            negative_id_c = negative_id.copy() 
            if sample_num == 1:
                if lb_i == 1 :
                    close_id2 = np.random.choice(positive_id_c,2,replace=False)
                    if close_id2[0] != i :
                        close_id = close_id2[0] 
                    else :
                        close_id = close_id2[1]  
                    away_id = np.random.choice(negative_id,1) 
                elif lb_i == 0 :
                    close_id2 = np.random.choice(negative_id_c,2,replace=False)
                    if close_id2[0] != i :
                        close_id = close_id2[0] 
                    else :
                        close_id = close_id2[1]  
                    away_id = np.random.choice(positive_id,1)
            else :
                if lb_i == 1 :
                    close_id = np.random.choice(positive_id_c,sample_num,replace=False)
                    away_id = np.random.choice(negative_id,sample_num,replace=False)[close_id!=i]
                    close_id = close_id[close_id!=i]
                elif lb_i == 0 :
                    close_id = np.random.choice(negative_id_c,sample_num,replace=False)
                    away_id = np.random.choice(positive_id,sample_num,replace=False)[close_id!=i]
                    close_id = close_id[close_id!=i]
                
            #print(close_id, away_id)
            if sample_num >1 :
                graph_rep_i, seq_rep_i= graph_rep[i].repeat(close_id.shape[0]), seq_rep[i].repeat(close_id.shape[0])
                prediction_emb_i = prediction_emb[i].repeat(close_id.shape[0]) 
            else :
                graph_rep_i, seq_rep_i= graph_rep[i], seq_rep[i]
                prediction_emb_i = prediction_emb[i]
            graph_rep_close, seq_rep_close= torch.squeeze(graph_rep[close_id]).view((-1,)), torch.squeeze(seq_rep[close_id]).view((-1,))
            graph_rep_away, seq_rep_away= torch.squeeze(graph_rep[away_id]).view((-1,)), torch.squeeze(seq_rep[away_id]).view((-1,))
            prediction_emb_close, prediction_emb_away = torch.squeeze(prediction_emb[close_id]).view((-1,)),torch.squeeze(prediction_emb[away_id]).view((-1,))  
            loss_1 = torch.log( torch.sigmoid( torch.clip( torch.dot(graph_rep_i, seq_rep_close) - torch.dot(graph_rep_i, seq_rep_away), min=-70 ) ) ) #permutation
            loss_2 = torch.log( torch.sigmoid( torch.clip(torch.dot(seq_rep_i, graph_rep_close) - torch.dot(seq_rep_i, graph_rep_away), min=-70 ) ) )
            loss_level_1 += torch.log( torch.sigmoid( torch.clip(torch.dot(prediction_emb_i, prediction_emb_close) - torch.dot(prediction_emb_i, prediction_emb_away), min=-70 ) ) ) 
            loss_level_2 += loss_1 + loss_2
            #print(loss) 
            
        if typ == "sum" :
            return loss_level_2, loss_level_1
"""