import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch
import pandas as pd
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.configs = configs

        # Decompsition Kernel Size
        kernel_size = configs.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)


        if self.configs.include_pred==1:
            self.project_dict = self.configs.project_dict
            self.num_new_col = len(self.project_dict)
            self.add_channel = self.num_new_col
            self.decompsition_pred = series_decomp(self.configs.kernel_size)
            self.lin1_enc=torch.nn.Linear(2*(self.configs.seq_len+self.configs.pred_len),self.configs.seq_len)
            self.revin_layer_enc = RevIN(self.configs.enc_in+10)
            
    def forward(self, x):
        
        if self.configs.include_pred==1:
            zeros = x[:, -self.configs.pred_len:, :]
            x_pred = torch.cat((x, zeros), dim=1)
            k=0
            for key, value in self.project_dict.items():
                k=k+1
                proj_to = value[0]-1 
                proj_from = value[1]-1 
                data = x[:, -1, proj_from:proj_from + self.configs.pred_len]
                x_pred[:, -self.configs.pred_len:, proj_to] = data
                x_pred[:, :self.configs.seq_len, -self.configs.c_out-k]=x_pred[:, -self.configs.seq_len:, proj_to]
                x_pred[:, -self.configs.pred_len:, -self.configs.c_out-k]=data

            x = x_pred[: , : , -self.configs.c_out-self.num_new_col:]
            x = self.revin_layer_enc(x, 'norm')
            seasonal_init, trend_init = self.decompsition(x)
            x= torch.cat([seasonal_init, trend_init], dim=1)
            x = torch.permute(x, (0,2,1))
            x = self.lin1(x)
            x = torch.permute(x, (0,2,1))
            x = self.revin_layer_enc(x, 'denorm')

        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
