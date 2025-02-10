import dgl
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph #pyg instead?


GCN_msg = dgl.function.copy_u( 'feat', 'message' ) #from src to out
GCN_reduce =  dgl.function.sum( msg='message', out='feat' )

class NodeApplyModule(nn.Module): #全连接层
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.activation = activation
        
    def forward(self,node):
        h = self.linear(node.data["feat"])
        if self.activation is not None:
            h = self.activation(h)
        return {'feat': h } 


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN,self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        
    def forward(self, g, feature):
        g.ndata["feat"] = feature
        g.update_all( GCN_msg, GCN_reduce )
        g.apply_nodes( func = self.apply_mod )
        return g.ndata.pop( "feat" )
    
    
class GraphNet(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphNet,self).__init__()
        self.num_nodes = 54012
        self.embed = nn.Parameter(torch.Tensor(self.num_nodes, in_feats), requires_grad = True)
        nn.init.xavier_uniform_( self.embed, gain=nn.init.calculate_gain('relu') ) 
        
        self.gcn1=GCN( in_feats, 32, F.relu)
        self.gcn2=GCN( 32, out_feats, None )
        self.gcnfc = nn.Linear( out_feats*2, out_feats )
        self.act = F.relu #activatin
        nn.init.xavier_uniform_(self.gcnfc.weight, gain=nn.init.calculate_gain('relu'))
        

    def forward(self, g, gene1:list, gene2:list):
        features = self.embed[ g.ndata['idx'] ]
        x=self.gcn1(g, features)
        x=self.gcn2(g, x)
        g.ndata['feat'] = x #pop之后不存在feat
        g_li = dgl.unbatch(g)
        gene1_feats = torch.cat( [ g_li[graph_id].ndata['feat'][g_li[graph_id].ndata['sl_gene_1']] for graph_id, gene_id in enumerate(gene1) ], dim=0 )
        gene2_feats = torch.cat( [ g_li[graph_id].ndata['feat'][g_li[graph_id].ndata['sl_gene_2']] for graph_id, gene_id in enumerate(gene2) ], dim=0 )
        x=self.gcnfc( torch.cat([gene1_feats, gene2_feats], dim=1) ) #将基因对feature concat
        x=self.act(x)
        return x #output batch*out_feats
     
    
def xavier_init(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)



class MLP(nn.Module):
    def __init__(self, input_size, projection_size, hid_size1=512, hid_size2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hid_size1),
            nn.BatchNorm1d(hid_size1), #不支持单个数据点
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size1, hid_size2),
            nn.BatchNorm1d(hid_size2),
            nn.LeakyReLU(inplace=True),
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(hid_size2, projection_size)
        ) #emb2score
        
        #self.sig = nn.Sigmoid() 
        self.net.apply(xavier_init)
        self.net2.apply(xavier_init)

    def forward(self, x):
        return self.net2(self.net(x)), self.net(x)
   
   
    
    
class CLNet(nn.Module):
     def __init__(self, input_size, projection_size, ablation=None):
        super().__init__()
        self.GraphEncoder = GraphNet( input_size, projection_size )
        self.SeqEncoder = MLP( input_size*2, projection_size )
        self.sig = nn.Sigmoid() 
        self.ablation = ablation
        if self.ablation is None:
            self.predictior = MLP( projection_size*2, 1, hid_size1=projection_size, hid_size2=int(projection_size/2) ) 
        else:
            self.predictior = MLP( projection_size, 1, hid_size1=projection_size, hid_size2=int(projection_size/2) ) 
          
     def forward(self, graph, seq1, seq2, gene1, gene2):
        graph_rep = self.GraphEncoder(graph, gene1, gene2)
        seq_rep, _ = self.SeqEncoder( torch.cat([seq1, seq2], dim=1) )
        
        if self.ablation is None:
            prediction_score, prediction_emb = self.predictior( torch.cat([graph_rep, seq_rep], dim=1) )
        elif self.ablation == 'graph_only' : 
            prediction_score, prediction_emb = self.predictior( graph_rep )
        elif self.ablation == 'seq_only' : 
            prediction_score, prediction_emb = self.predictior( seq_rep )
            
        prediction_score = self.sig (prediction_score)
        return graph_rep, seq_rep, prediction_score, prediction_emb