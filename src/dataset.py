from torch.utils.data import Dataset
import numpy as np
import dgl
import torchvision.transforms as transforms
import torch
class CLDataset(Dataset):
    def __init__(self, KG_fd:str, SL_pair_pth:list, SL_lb_pth:list, mode='train', isomorphic=True, emb='esm2', max_neighbour=200, subgraph_hop=1, train_ratio=1, test_nr_rate=None ):
        self.mode = mode; self.isomorphic = isomorphic; self.emb = emb; self.max_neighbour = max_neighbour; self.subgraph_hop = subgraph_hop
        if self.mode == 'train':
            self.data_pairs = np.load(SL_pair_pth[0], allow_pickle=True)[0]
            self.pair_lbs = np.load(SL_lb_pth[0], allow_pickle=True)[0]
            train_indices = np.random.choice( self.data_pairs.shape[0], size=int(train_ratio*self.data_pairs.shape[0]), replace=False)
            self.data_pairs = self.data_pairs[train_indices]
            self.pair_lbs = self.pair_lbs[train_indices]
            print(f"train ratio is {train_ratio}, {len(train_indices)} pairs")
        else:
            if self.mode == 'valid':
                self.data_pairs = np.load(SL_pair_pth[1], allow_pickle=True)[0]
                self.pair_lbs = np.load(SL_lb_pth[1], allow_pickle=True)[0] 
            elif self.mode == 'test':
                self.data_pairs = np.load(SL_pair_pth[2], allow_pickle=True)[0]
                self.pair_lbs = np.load(SL_lb_pth[2], allow_pickle=True)[0]   #data多了一个dim  
            if test_nr_rate is not None:
                pos_idx = np.where(self.pair_lbs==1)[0] 
                neg_num = int( pos_idx.shape[0] * test_nr_rate )
                neg_idx = np.random.choice( np.where(self.pair_lbs==0)[0], neg_num )
                idx = np.hstack( (pos_idx, neg_idx) )
                self.data_pairs = self.data_pairs[idx]
                self.pair_lbs = self.pair_lbs[idx]
                #print(self.data_pairs.shape)
                #print(self.pair_lbs.shape)
                
        test_msk = np.load(SL_pair_pth[2], allow_pickle=True)
        test_lb = np.load(SL_lb_pth[2], allow_pickle=True) 
        test_msk = test_msk[test_lb==1] #sl pairs should be masked in KG
        valid_msk = np.load(SL_pair_pth[1], allow_pickle=True)
        valid_lb = np.load(SL_lb_pth[1], allow_pickle=True) 
        valid_msk = valid_msk[valid_lb==1]
        self.masked_pairs = np.vstack((test_msk,valid_msk)).tolist()

        self.id2esmemb = dict( np.load(f"{KG_fd}/kgid2esmemb.npy", allow_pickle=True).tolist() )
        
        if self.isomorphic:
            KG_triple = np.load(f"{KG_fd}/KG_triple_isomorphic.npy", allow_pickle=True).tolist()
            for pair in self.masked_pairs:
                try:
                    KG_triple.remove(pair)
                except:
                    pass
                try:
                    KG_triple.remove([pair[1],pair[0]])
                except:
                    pass
            src_ids=[ t[0] for t in KG_triple ]
            dst_ids=[ t[1] for t in KG_triple ]
            self.KG = dgl.graph((src_ids, dst_ids))  #gcn不考虑边的种类
            self.KG.ndata['idx'] = torch.tensor([ i for i in range(54012)])
        else:
            pass
        
        self.__getitem__(0)
        
        
    def __getitem__( self, index ):
        gene1, gene2 = self.data_pairs[index] #file中 data多了一个dimension
        sl_label = self.pair_lbs[index]
        subgraph_nodes = [gene1, gene2]; 
        for i in range(self.subgraph_hop):
            gene_KG = dgl.sampling.sample_neighbors( self.KG, subgraph_nodes, self.max_neighbour )
            us = gene_KG.edges()[0].tolist()
            vs = gene_KG.edges()[1].tolist()
            subgraph_nodes = list(set(vs + us))
        subgraph_nodes = list( set( subgraph_nodes + [gene1, gene2] ) )   #gene1或者和gene2可能没边
        gene_subgraph = dgl.node_subgraph( self.KG, subgraph_nodes)
        gene_subgraph.ndata['sl_gene_1'] = gene_subgraph.ndata['idx'] == gene1  #gene1 and gene2 mask
        gene_subgraph.ndata['sl_gene_2'] = gene_subgraph.ndata['idx'] == gene2
        if self.emb == 'esm2':
            gene1_emb = self.id2esmemb[gene1]
            gene2_emb = self.id2esmemb[gene2] 
            self.emb_size = gene2_emb.shape[0]
        
        return gene_subgraph, gene1_emb, gene2_emb,sl_label, gene1, gene2
    
    
    def __len__(self):
        return self.pair_lbs.shape[0]