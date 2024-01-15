import json
import torch as t
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data, seed_all
from FGANomaly import FGANomalyModel, RNNAutoEncoder, MLPDiscriminator

from TSds import TSds

import os
import numpy as np 
import argparse

import pandas as pd



if __name__ =="__main__":
    
    #read the parser

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type = str, required=True)
    parser.add_argument('--WL',type=int, required=True)
    parser.add_argument('--n',type=int, required=True)
    parser.add_argument('--i', type=int, required = True)

    args = parser.parse_args()

    print(args)
    
    #Reading the params of the model
    with open("params.json", "r") as file:

         params = json.load(file)

    #'data_prefix': 'msl',
    #'best_model_path': os.path.join('rnn_output', 'best_model'),
    #'result_path': os.path.join('rnn_output'),
    #'device': t.device('cuda:3' if t.cuda.is_available() else 'cpu'), 
    ds = TSds.read_UCR(args.path)
    
    params['window_size'] = args.WL * args.n
    params['data_prefix'] = ds.name
    params['best_model_path'] = f'best_models/{params["data_prefix"]}/{args.WL*args.n}'
    params['result_path'] = f'results/{params["data_prefix"]}'
    params['device'] = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'Using device:{params["device"]}')

    data = load_data(train_x = ds.ts_scaled[:ds.train_split],
                 test_x = ds.ts_scaled[ds.train_split:],
                 test_y = np.array(ds.df['is_anomaly'][ds.train_split:]),
                 val_size=0.2,
                 window_size = params['window_size']
                 dataloader=True)
    data['nc'] = 1

    model = FGANomalyModel(ae=RNNAutoEncoder(inp_dim=data['nc'],
                                            z_dim=params['z_dim'],
                                            hidden_dim=params['hidden_dim'],
                                            rnn_hidden_dim=params['rnn_hidden_dim'],
                                            num_layers=params['num_layers'],
                                            bidirectional=params['bidirectional'],
                                            cell=params['cell']),

                            dis_ar=MLPDiscriminator(inp_dim=data['nc'],
                            hidden_dim=params['hidden_dim']),
                            data_loader=data, **params)

    model.train()
    res = model.test()
    
    res['dataset'] = params['data_prefix']
    res['WL'] = args.WL 
    res['n'] = args.n
    res['id'] = args.i

    fileName = "res_FGAN_UCR_falta.csv"
    
    if os.path.exists(fileName):
        print('existe')
        df = pd.read_csv(fileName)
        df = pd.concat([df, pd.DataFrame([res])]).to_csv(fileName, index = False)

    else:

        pd.DataFrame([res]).to_csv(fileName, index = False)
    print(res)

