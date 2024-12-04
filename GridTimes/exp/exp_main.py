from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TimeMachine, GridTimes, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, iTransformer , Informer, TimesNet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import pandas as pd
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.df_dict = self.args.col_info_dict

    def _build_model(self):
        model_dict = {
            'TimeMachine':TimeMachine,
            'GridTimes': GridTimes,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimesNet': TimesNet,
            
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        
        loss_sv = {}
        total_loss_sv = {}
        
        for key in self.df_dict:
            loss_sv[key] = []
            total_loss_sv[key] = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                if self.args.include_pred == 0:
                    batch_x = batch_x[: , : , -22:].float().to(self.device)
                    batch_y = batch_y[: , : , -22:].float()
                else:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model or 'Grid' in self.args.model or 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)                        
                        
                else:
                    if 'Machine' in self.args.model or 'Grid' in self.args.model or 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                    
                if self.args.features == 'MS':
                    f_dim = -1
                elif self.args.features == 'Mm':
                    f_dim = -self.args.n_pred_col
                else: 
                    f_dim = 0
                    
                outputs_ = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_ = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs_.detach().cpu()
                true = batch_y_.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)


                for key, val in self.df_dict.items():
                    if val[0]+val[1] == 0:
                        outputs_ = outputs[:, -self.args.pred_len:, val[0]:]
                        batch_y_ = batch_y[:, -self.args.pred_len:, val[0]:].to(self.device)
                    else:
                        outputs_ = outputs[:, -self.args.pred_len:, val[0]:val[0]+val[1]]
                        batch_y_ = batch_y[:, -self.args.pred_len:, val[0]:val[0]+val[1]].to(self.device)
                    
                    pred = outputs_.detach().cpu()
                    true = batch_y_.detach().cpu()
                    loss = criterion(pred, true)
                    loss_sv[key].append(loss)

        total_loss = np.average(total_loss)
        for key, val in self.df_dict.items():
            total_loss_sv[key] = np.average(loss_sv[key])

        self.model.train()
        return total_loss , total_loss_sv

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        total_params = sum(p.numel() for p in self.model.parameters())
        print('#total parameters:' , total_params)
        for epoch in range(self.args.train_epochs):
        
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.include_pred == 0:
                    batch_x = batch_x[: , : , -22:].float().to(self.device)
                    batch_y = batch_y[: , : , -22:].float().to(self.device)
                else:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model or 'Grid' in self.args.model or 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        

                        # f_dim = -1 if self.args.features == 'MS' else 0
                        if self.args.features == 'MS':
                            f_dim = -1
                        elif self.args.features == 'Mm':
                            f_dim = -self.args.n_pred_col
                        else: 
                            f_dim = 0
                            
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Machine' in self.args.model or 'Grid' in self.args.model or 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                            
                   
                    if self.args.features == 'MS':
                        f_dim = -1
                    elif self.args.features == 'Mm':
                        f_dim = -self.args.n_pred_col
                    else: 
                        f_dim = 0

                    outputs_ = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_ = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs_, batch_y_)
                    train_loss.append(loss.item())
                                    
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss , vali_loss_sv = self.vali(vali_data, vali_loader, criterion)
            test_loss , test_loss_sv = self.vali(test_data, test_loader, criterion)
            
            print(f"Epoch: {epoch + 1}/{self.args.train_epochs}, Train Loss: {train_loss:.7f}, Validation Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            for key, val in self.df_dict.items():
                print(f'FOR {key}:average loss is {test_loss_sv[key]}')
            if 'TST' in self.args.model:
                if epoch>0:
                    early_stopping(vali_loss, self.model, path)
            else:
                early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        
        preds_sv = {}
        trues_sv = {}
        for key in self.df_dict:
            preds_sv[key] = []
            trues_sv[key] = []
            
        # preds_er = {}
        # trues_er = {}
        # ercot_er = {}
        # for key in self.args.project_dict:
        #     preds_er[key] = []
        #     trues_er[key] = []
        #     ercot_er[key] = []
        
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                if self.args.include_pred == 0:
                    ercot_data_x = batch_x[: , : , : -22]
                    batch_x = batch_x[: , : , -22:]
                    batch_x = batch_x.float().to(self.device)
                    
                    ercot_data_y = batch_y[: , : , : -22]
                    batch_y = batch_y[: , : , -22:]
                    batch_y = batch_y.float().to(self.device)
                else:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model or 'Grid' in self.args.model or 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                else:
                    if 'Machine' in self.args.model or 'Grid' in self.args.model or 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                
                if self.args.features == 'MS':
                    f_dim = -1
                elif self.args.features == 'Mm':
                    f_dim = -self.args.n_pred_col
                else: 
                    f_dim = 0
                    
                outputs_ = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_ = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs_.detach().cpu().numpy()
                true = batch_y_.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    prd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, prd, os.path.join(folder_path, str(i) + '.pdf'))

                for key, val in self.df_dict.items():
                    if val[0]+val[1] == 0:
                        outputs_ = outputs[:, -self.args.pred_len:, val[0]:]
                        batch_y_ = batch_y[:, -self.args.pred_len:, val[0]:].to(self.device)
                    else:
                        outputs_ = outputs[:, -self.args.pred_len:, val[0]:val[0]+val[1]]
                        batch_y_ = batch_y[:, -self.args.pred_len:, val[0]:val[0]+val[1]].to(self.device)
                        
                    pred = outputs_.detach().cpu().numpy()
                    true = batch_y_.detach().cpu().numpy()

                    preds_sv[key].append(pred)
                    trues_sv[key].append(true)
                    
                # for key, val in self.args.project_dict.items():
                #     ercot = (ercot_data_y[: , -self.args.pred_len-1 , val[1]-1:val[1]-1+self.args.pred_len]).detach().cpu().numpy()
                #     true = (batch_y[:, -self.args.pred_len:, val[0]-1-240].to(self.device)).detach().cpu().numpy()
                #     pred = (outputs[:, -self.args.pred_len:, val[0]-1-240]).detach().cpu().numpy()

                #     preds_er[key].append(pred)
                #     trues_er[key].append(true)
                #     ercot_er[key].append(ercot)
                #     print('key: ' , key)
                #     print(ercot[0 , :])
                #     print(true[0 , :])
                #     print(pred[0 , :])
                    
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        inputx = np.array(inputx)
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)                

        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        temp_df = pd.DataFrame()
        temp_df['Seed']=[self.args.random_seed]
        temp_df['Model']=[self.args.model]
        temp_df['seq_len']=[self.args.seq_len]
        temp_df['label_len']=[self.args.label_len]
        temp_df['pred_len']=[self.args.pred_len]
        temp_df['train_epochs']=[self.args.train_epochs]
        temp_df['batch']=[self.args.batch_size]
        temp_df['LR']=[self.args.learning_rate]
        temp_df['top_k']=[self.args.top_k]
        temp_df['d_ff']=[self.args.d_ff]
        temp_df['e_layers']=[self.args.e_layers]
        temp_df['n_heads']=[self.args.n_heads]
        temp_df['d_model']=[self.args.d_model]
        temp_df['fc_drop']=[self.args.dropout]
        temp_df['n1']=[self.args.n1]
        temp_df['n2']=[self.args.n2]
        temp_df['n_embed']=[self.args.n_embed]
        temp_df['kernel_size']=[self.args.kernel_size]
        temp_df['lradj']=[self.args.lradj]

        temp_df['MSE']=[mse]
        temp_df['MAE']=[mae]


        if not os.path.exists('./csv_results/'+'result_'+self.args.data_path):
            temp_df.to_csv('./csv_results/'+'result_'+self.args.data_path, index=False)
        else:
            result_df=pd.read_csv('./csv_results/'+'result_'+self.args.data_path)
            result_df = pd.concat([result_df,temp_df],ignore_index=True)
            result_df.to_csv('./csv_results/'+'result_'+self.args.data_path, index=False)

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)

        for key , val in self.df_dict.items():
            preds_tmp = np.array(preds_sv[key])
            trues_tmp = np.array(trues_sv[key])
            preds_tmp = preds_tmp.reshape(-1, preds_tmp.shape[-2], preds_tmp.shape[-1])
            trues_tmp = trues_tmp.reshape(-1, trues_tmp.shape[-2], trues_tmp.shape[-1])
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds_tmp, trues_tmp)
            print(f'FOR {key}: mse:{mse}, mae:{mae}, rse:{rse}')
        # print('comparing with ERCOT: ')
        
        # for key , val in self.args.project_dict.items():
        #     preds_tmp = np.array(preds_er[key])
        #     trues_tmp = np.array(trues_er[key])
        #     ercot_tmp = np.array(ercot_er[key])
            
        #     preds_tmp = preds_tmp.reshape(-1, preds_tmp.shape[-2], preds_tmp.shape[-1])
        #     trues_tmp = trues_tmp.reshape(-1, trues_tmp.shape[-2], trues_tmp.shape[-1])
        #     ercot_tmp = ercot_tmp.reshape(-1, ercot_tmp.shape[-2], ercot_tmp.shape[-1])
            
        #     mae, mse, rmse, mape, mspe, rse, corr = metric(preds_tmp, trues_tmp)
        #     print(f'For our model and {key}: mse:{mse}, mae:{mae}, rse:{rse}')
        #     mae, mse, rmse, mape, mspe, rse, corr = metric(ercot_tmp, trues_tmp)
        #     print(f'For ERCOT model and {key}: mse:{mse}, mae:{mae}, rse:{rse}')
            
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model or 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                       
                else:
                    if 'Machine' in self.args.model or 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
