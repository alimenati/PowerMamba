import torch
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN
import torch
import pandas as pd
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN
import torch.nn.functional as F

torch.cuda.empty_cache()

class moving_avg(torch.nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
        

class series_decomp(torch.nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(my_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(torch.nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.configs=configs
        if self.configs.include_pred==1:
            self.project_dict = self.configs.project_dict
            self.num_new_col = len(self.project_dict)
            self.add_channel = self.num_new_col
            self.configs.enc_in=self.configs.enc_in+self.add_channel
            self.decompsition_pred = series_decomp(self.configs.kernel_size)
            self.lin1_enc=torch.nn.Linear(2*(self.configs.seq_len+self.configs.pred_len),self.configs.seq_len)
            self.revin_layer_enc = RevIN(self.configs.enc_in)
            
        else:
            self.add_channel = 0
            
            
        self.revin_layer = RevIN(self.configs.enc_in)
        self.lin1=torch.nn.Linear(self.configs.seq_len,self.configs.n1)
        self.dropout1=torch.nn.Dropout(self.configs.dropout)

        self.lin2=torch.nn.Linear(self.configs.n1,self.configs.n2)
        self.dropout2=torch.nn.Dropout(self.configs.dropout)
        if self.configs.ch_ind==1:
            self.d_model_param1=1
            self.d_model_param2=1

        else:
            self.d_model_param1=self.configs.n2
            self.d_model_param2=self.configs.n1

        self.mamba1=Mamba(d_model=self.d_model_param1,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact) 
        self.mamba2=Mamba(d_model=self.configs.n2,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact) 
        self.mamba3=Mamba(d_model=self.configs.n1,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)
        self.mamba4=Mamba(d_model=self.d_model_param2,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)

        self.lin3=torch.nn.Linear(self.configs.n2,self.configs.n1)
        self.lin4=torch.nn.Linear(2*self.configs.n1,self.configs.pred_len)

            
            
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
             seasonal_init, trend_init = self.decompsition_pred(x)
             x= torch.cat([seasonal_init, trend_init], dim=1)
             x = torch.permute(x, (0,2,1))
             x = self.lin1_enc(x)
             x = torch.permute(x, (0,2,1))
             x = self.revin_layer_enc(x, 'denorm')

         if self.configs.revin==1:
             x=self.revin_layer(x,'norm')
         else:
             means = x.mean(1, keepdim=True).detach()
             x = x - means
             stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
             x = x / stdev
         
         x=torch.permute(x,(0,2,1))
         if self.configs.ch_ind==1:
             x=torch.reshape(x,(x.shape[0]*x.shape[1],1,x.shape[2]))

         x=self.lin1(x)
         x_res1=x
         x=self.dropout1(x)
         x3=self.mamba3(x)
         if self.configs.ch_ind==1:
             x4=torch.permute(x,(0,2,1))
         else:
             x4=x
         x4=self.mamba4(x4)
         if self.configs.ch_ind==1:
             x4=torch.permute(x4,(0,2,1))
       
         x4=x4+x3

         x=self.lin2(x)
         x_res2=x
         x=self.dropout2(x)
         
         if self.configs.ch_ind==1:
             x1=torch.permute(x,(0,2,1))
         else:
             x1=x      
         x1=self.mamba1(x1)
         if self.configs.ch_ind==1:
             x1=torch.permute(x1,(0,2,1))
  
         x2=self.mamba2(x)

         if self.configs.residual==1:
             x=x1+x_res2+x2
         else:
             x=x1+x2
         
         x=self.lin3(x)
         if self.configs.residual==1:
             x=x+x_res1
             
         x=torch.cat([x,x4],dim=2)
         x=self.lin4(x) 
         if self.configs.ch_ind==1:
             x=torch.reshape(x,(-1,self.configs.enc_in,self.configs.pred_len))
         
         x=torch.permute(x,(0,2,1))
         if self.configs.revin==1:
             x=self.revin_layer(x,'denorm')
         else:
             x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
             x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        

         return x
        
