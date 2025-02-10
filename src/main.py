import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import dgl
import torch 
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph 
import argparse
from dataset import CLDataset
from model import CLNet
from trainer import Trainer
from evaluator import Evaluator
import logging
import time
import gc 
import traceback 
import numpy as np
gc.collect()



if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.INFO)
        
        parser = argparse.ArgumentParser(description='CLSL model')
        #parser.add_argument("--seed", type=int, default=88, help="Which seed to use?")

        parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")

        parser.add_argument("--optimizer", type=str, default="Adam", help="which optimizer to use")
        parser.add_argument("--scheduler_step", type=int, default=10, help="step size of the scheduler")
        parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="gamma of the scheduler")
        parser.add_argument("--lr", type=float, default=5e-3, help="learning rate of the optimizer")
        parser.add_argument("--clip", type=int, default=1000, help="maximum gradient norm allowed")

        parser.add_argument("--train_ratio", type=float, default=1, help="rate of training data")
        parser.add_argument("--l2", type=float, default=1e-5, help="regularization constant for GNN weights")
        #parser.add_argument("--loss_cl", type=float, default=5e-4, help="weights for cl_loss")
        parser.add_argument("--loss_cl_level_2", type=float, default=1e-9, help="weights for cl_loss_level_2")
        parser.add_argument("--loss_cl_level_1", type=float, default=5e-2, help="weights for cl_loss_level_1")
        parser.add_argument("--loss_cl_sample", type=int, default=1, help="how many samples when calculating cl_loss")
        
        parser.add_argument("--test_nr_rate", type=int, default=None, help="n/p rate of data in test set")
        
        parser.add_argument('--momentum', type=float, default=0.5, help='momentum for target encoder optimization')
        parser.add_argument("--batch_size", type=int, default=16, help="batch size")
        parser.add_argument("--num_epochs", type=int, default=15, help="epoch")
        parser.add_argument("--model_criterion", type=str, default='aupr', help="criterion to update the model")
        
        parser.add_argument("--result_dir", type=str, default='../result', help="where is the result")
        parser.add_argument("--expnm", type=str, default='cell_line_fix_test_median_b', help="experiment name")
        parser.add_argument("--data_dir", type=str, default='../data', help="where is the data")
        parser.add_argument("--cell_line", type=str, default='A549', help="cell_line")
        parser.add_argument("--store_model", default=False, action='store_true', help='model should be stored or not')
        
        parser.add_argument("--wo_cl_loss_level_2", default=False, action='store_true', help='whether filter cl loss level2 or not')
        parser.add_argument("--wo_cl_loss_level_1", default=False, action='store_true', help='whether filter cl loss level1 or not')
        parser.add_argument("--ablation", type=str, default=None, help="params for the ablation study, graph_only or seq_only")
        
        params = parser.parse_known_args()[0]
        
        #np.random.seed(params.seed)
        
        if params.cell_line == 'RPE1':
            nr_li = [None]           #nr_rate has been changed
        else:
            nr_li = [1.0]  
            
        for nr_rate in nr_li:

            if params.wo_cl_loss_level_1 and params.wo_cl_loss_level_2 :
                print("Training w/o cl_loss")
                params.result_dir = f'{params.result_dir}_woCL'
                params.expdir = f'{params.expnm}_tr{params.train_ratio}_trainnr{nr_rate}_testnr{params.test_nr_rate}'
            elif (not params.wo_cl_loss_level_1) and (not params.wo_cl_loss_level_2)  :
                print("Training wiz cl_loss")
                params.expdir = f'{params.expnm}_cl1{params.loss_cl_level_1}_cl2{params.loss_cl_level_2}_sample{params.loss_cl_sample}_l2{params.l2}_tr{params.train_ratio}_trainnr{nr_rate}_testnr{params.test_nr_rate}'
            elif  params.wo_cl_loss_level_1 and (not params.wo_cl_loss_level_2)  : 
                params.expdir = f'{params.expnm}_wocl1_cl2{params.loss_cl_level_2}_sample{params.loss_cl_sample}_l2{params.l2}_tr{params.train_ratio}_trainnr{nr_rate}_testnr{params.test_nr_rate}'
            elif (not params.wo_cl_loss_level_1) and params.wo_cl_loss_level_2 : 
                params.expdir = f'{params.expnm}_cl1{params.loss_cl_level_1}_wocl2_sample{params.loss_cl_sample}_l2{params.l2}_tr{params.train_ratio}_trainnr{nr_rate}_testnr{params.test_nr_rate}'    
                
            if not os.path.exists(f'{params.result_dir}'):
                os.mkdir(f'{params.result_dir}')                 
            if not os.path.exists(f'{params.result_dir}/{params.expdir}'):
                os.mkdir(f'{params.result_dir}/{params.expdir}')
            if not os.path.exists(f'{params.result_dir}/{params.expdir}/{params.cell_line}'):
                os.mkdir(f'{params.result_dir}/{params.expdir}/{params.cell_line}')
                
            if torch.cuda.is_available():
                print("using cuda")
                params.device = torch.device('cuda:%d' % params.gpu)
            else:
                params.device = torch.device('cpu')
                
            
            str_time = time.strftime('%y-%m-%d %H:%M:%S')
            file_handler = logging.FileHandler(f'{params.result_dir}/{params.expdir}/{params.cell_line}/log_{str_time}.txt', mode='a')
           
                 
            logger = logging.getLogger()
            logger.addHandler(file_handler)
            logger.info('============ Initialized logger ============')
            logger.info('\t '.join('%s: %s' % (k, str(v)) for k, v
                                in sorted(dict(vars(params)).items())))
            logger.info('============================================')
            
            logger.info(f'******Experiments in {params.expnm} in cell-line {params.cell_line} with nr_rate {nr_rate}******') 
            sl_pths = [f'{params.data_dir}/{params.expnm}/{params.cell_line}/gene_split_c1_nr_{nr_rate}/train.npy', f'{params.data_dir}/{params.expnm}/{params.cell_line}/gene_split_c1_nr_{nr_rate}/valid.npy', f'{params.data_dir}/{params.expnm}/{params.cell_line}/gene_split_c1_nr_{nr_rate}/test.npy']
            sl_lb_pths = [f'{params.data_dir}/{params.expnm}/{params.cell_line}/gene_split_c1_nr_{nr_rate}/train_lbs.npy', f'{params.data_dir}/{params.expnm}/{params.cell_line}/gene_split_c1_nr_{nr_rate}/valid_lbs.npy', f'{params.data_dir}/{params.expnm}/{params.cell_line}/gene_split_c1_nr_{nr_rate}/test_lbs.npy']  
            
            train = CLDataset('../data', sl_pths, sl_lb_pths, mode='train', train_ratio=params.train_ratio)  
            valid = CLDataset('../data', sl_pths, sl_lb_pths, mode='valid', test_nr_rate=params.test_nr_rate )  
            test = CLDataset('../data', sl_pths, sl_lb_pths, mode='test', test_nr_rate=params.test_nr_rate)  
            print(f"The original embedding size is {train.emb_size}.")
            
            for trial in [1,2,3,4,5] :
                clsl = CLNet( train.emb_size, 16, ablation=params.ablation )
                valid_evaluator = Evaluator(params, clsl, valid )
                test_evaluator = Evaluator(params, clsl, test )
                trainer = Trainer( params, clsl, train_data=train, valid_evaluator=valid_evaluator, test_evaluator=test_evaluator, nr_rate=nr_rate, result_suffix=trial )
                trainer.train()
            
    except:
        # 返回时间和原始报错信息
        logging.error(time.strftime('%y-%m-%d %H:%M:%S')+traceback.format_exc() + '-------------- \n')
