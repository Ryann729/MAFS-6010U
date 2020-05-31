# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:34:12 2020

@author: ZHONG Jing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from sklearn.preprocessing import MinMaxScaler
import time

## arima
import pmdarima as pm
from pmdarima import model_selection
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
import statsmodels as sm

## LSTM
import torch
import torch.nn as nn
from torch.autograd import Variable
from fastprogress import progress_bar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
from collections import Iterable
def flattern(a):
    for each in a:
        if not isinstance(each, Iterable) or isinstance(each, str):
            yield each 
        else:
            yield from flattern(each)
            
## evaluation metrics
def MSE(actual: np.array, pred: np.array):
    return( np.sqrt(((actual - pred) ** 2).mean()) )

def RMSSE(actual: np.array, pred: np.array):
    '''
    Description:
    actual is the true value on both train and test data
    pred is the model prediction on test data
    '''
    return( np.sqrt(((actual[-len(pred):] - pred) ** 2).mean() / (np.diff(actual[:-len(pred)]) ** 2).mean()) )

###############################################################################
## load data
dataPath = 'C:\\Users\\13269\\Desktop\\20 Spring Semester\\MAFS6010U Artificial Intelligence in Finance\\project\\m5-forecasting-accuracy\\'
calendar = pd.read_csv(dataPath+'calendar.csv')
sales_train_validation = pd.read_csv(dataPath+'sales_train_validation.csv')
sell_prices = pd.read_csv(dataPath+'sell_prices.csv')

############################# data cleaning ###################################
## calendar
DF_calendar = calendar[['date','wm_yr_wk','wday','month','year','event_name_1','event_type_1','snap_CA','snap_TX','snap_WI']].copy(deep=True)
DF_calendar['holiday'] = -(DF_calendar.event_name_1.isnull() - 1)
DF_calendar['date'] = pd.to_datetime(DF_calendar['date'])

#Create date index
dates = DF_calendar['date'][0:1913] 

## sales_train_validation
sales_train_validation = sales_train_validation.assign(id=sales_train_validation.id.str.replace("_validation", ""))
DF_Sales = sales_train_validation.loc[:,'d_1':'d_'+str(len(dates))].T
# Set id as columns, date as index 
DF_Sales.columns = sales_train_validation['id'].values
DF_Sales = DF_Sales.set_index(dates)
id_all = DF_Sales.columns

## sell_prices
DF_sell_prices = sell_prices.copy(deep=True)
DF_sell_prices['id'] = DF_sell_prices['item_id'] + '_' + DF_sell_prices['store_id']
DF_sell_prices['state_id'] = DF_sell_prices['store_id'].str[:2]

## compute state average price of each item
group_state = DF_sell_prices.groupby(['wm_yr_wk', 'item_id', 'state_id'])
state_avg_sell_price = group_state['sell_price'].agg('mean').reset_index()
state_avg_sell_price = state_avg_sell_price.pivot_table(index=['wm_yr_wk','item_id'], columns=['state_id'], values='sell_price')
state_avg_sell_price.columns = list(map(lambda x: 'sell_price_' + x, state_avg_sell_price.columns))

DF_sell_prices = pd.merge(DF_sell_prices, state_avg_sell_price.reset_index(), on=['wm_yr_wk','item_id'])  
    
del calendar, sales_train_validation, sell_prices

############################################################################### 
## modeling for one item
## here take i_id = 100 as example
i_id = 100

state_id = id_all[i_id][-4:-2]
store_id = id_all[i_id][-4:]
item_id = id_all[i_id][:-5]
dept_id = item_id[:-4]
cat_id = dept_id [:-2]

## filter for different state or item
state = ['CA', 'TX', 'WI']
state.remove(state_id)
snap_del = list(map(lambda x: 'snap_' + x, state))

## merge data
model_data = pd.merge(DF_Sales[id_all[i_id]].reset_index(), DF_calendar, on='date', how='left')  
model_data = pd.merge(model_data, DF_sell_prices.loc[DF_sell_prices.id==id_all[i_id]], on='wm_yr_wk', how='left')
model_data = model_data.set_index(model_data.date)
model_data.drop(['date', 'wm_yr_wk', 'store_id', 'item_id', 'id'] + snap_del, axis=1, inplace=True)
series = model_data.iloc[:,0]

###############################################################################
## model 1: ARIMA
series_diff = series.diff().dropna()
plt.plot(series_diff)
## ADF stationary test(p-value 6.575445276313532e-27, stationary)
sm.tsa.stattools.adfuller(series_diff)

## Ljung-Box white noise test(p-value close to 0, not white noise)
plt.plot(lb_test(series_diff, lags=None, boxpierce=False)[1])
plt.show()    

## data split
train, test = model_selection.train_test_split(series, test_size=28)
## train model
arima_model = pm.auto_arima(train, trace=True, stepwise=True, suppress_warnings=True, error_action='ignore')
arima_model.summary()

## model checking(all p-value > 0.05, residual is white noise, model is correct)
plt.plot(lb_test(arima_model.resid(), lags=None, boxpierce=False)[1])
plt.axhline(y=0.05, c="r", ls="--", lw=2)
plt.show()    

## predict
preds, conf_int = arima_model.predict(n_periods=test.shape[0], return_conf_int=True)

## plot
## Entire Set
x_axis = model_data.index
plt.axvline(x_axis[-test.shape[0]], c='r')
plt.plot(x_axis, series)
plt.plot(x_axis, np.append(arima_model.predict_in_sample(),preds))  # Forecasts
plt.legend(['Prediction','Time Series'])
plt.fill_between(x_axis[-preds.shape[0]:], conf_int[:, 0], conf_int[:, 1], alpha=0.1, color='b')
plt.ylabel(ylabel = 'Sales Demand')
plt.xlabel(xlabel = 'Date')
plt.title('Time-Series Prediction Entire Set of ' + id_all[i_id])
plt.gcf().autofmt_xdate()
plt.show()

## Test Set
start_point = train.shape[0] - 2 * test.shape[0]
plt.axvline(x_axis[-test.shape[0]], c='r')
plt.plot(x_axis[start_point:], series[start_point:])
plt.plot(x_axis[start_point:], np.append(arima_model.predict_in_sample()[start_point:],preds))
ax = plt.gca() 
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d')) 
plt.xticks(pd.date_range(x_axis[start_point],x_axis[-1],freq='7d'))
plt.gcf().autofmt_xdate()
plt.legend(['Prediction','Time Series'])
plt.fill_between(x_axis[-preds.shape[0]:], conf_int[:, 0], conf_int[:, 1], alpha=0.1, color='b')
plt.ylabel(ylabel = 'Sales Demand')
plt.xlabel(xlabel = 'Date')
plt.title('Time-Series Prediction Test Set of ' + id_all[i_id])
plt.show()

## insample RMSE
MSE(series[:-len(test)], arima_model.predict_in_sample())
## outsample RMSE
MSE(series[-len(test):], preds)
## RMSSE
RMSSE(series, preds)

###############################################################################
## model 2: LSTM

##############################################
## data pre-processing
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

## data normalization
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(model_data[id_all[i_id]].values.reshape(-1, 1))

## set seq_length to look back
seq_length = 28
x, y = sliding_windows(train_data_normalized, seq_length)
print(x.shape)
print(y.shape)

## define test size
test_size = 28
train_size = len(y) - 2 * test_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))
trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
validX = Variable(torch.Tensor(np.array(x[train_size:(train_size+test_size)])))
validY = Variable(torch.Tensor(np.array(y[train_size:(train_size+test_size)])))
testX = Variable(torch.Tensor(np.array(x[(train_size+test_size):len(x)])))
testY = Variable(torch.Tensor(np.array(y[(train_size+test_size):len(y)])))

##############################################
## single layer
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.seq_length = seq_length
        self.dropout = nn.Dropout(p=0.2)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,dropout = 0.25)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        out = self.dropout(out)
       
        return out
    
#####  Parameters  ######################
num_epochs = 101
learning_rate = 1e-3
input_size = 1
hidden_size = 256
num_layers = 1
num_classes = 1

#####Init the Model #######################
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
lstm.to(device)

##### Set Criterion Optimzer and scheduler ####################
criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500, factor =0.5 ,min_lr=1e-7, eps=1e-08)

# Train the model
for epoch in progress_bar(range(num_epochs)): 
    lstm.train()
    outputs = lstm(trainX.to(device))
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY.to(device))
    loss.backward()
    optimizer.step()
    
    #Evaluate on validate     
    lstm.eval()
    valid = lstm(validX.to(device))
    vall_loss = criterion(valid, validY.to(device))
    scheduler.step(vall_loss)
    
    if epoch % 25 == 0:
      print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(), vall_loss.cpu().item()))
    
    ## deal with CUDA out of memory problem
    outputs = outputs.cpu()
    loss = loss.cpu()
    valid = valid.cpu()
    vall_loss = vall_loss.cpu()
    torch.cuda.empty_cache()
    
######Prediction###############
lstm.eval()
train_predict = lstm(dataX.to(device))
data_predict = train_predict.cpu().data.numpy()
dataY_plot = dataY.data.numpy()

## Inverse Normalize 
data_predict = scaler.inverse_transform(data_predict)
dataY_plot = scaler.inverse_transform(dataY_plot)

## Add dates
df_predict = pd.DataFrame(data_predict)
df_predict = df_predict.set_index([model_data.index[(1+seq_length):]])
df_labels = pd.DataFrame(dataY_plot)
df_labels = df_labels.set_index([model_data.index[(1+seq_length):]])

# Plot Entire Set 
plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
plt.axvline(x=model_data.index[-28], c='r')
plt.plot(df_labels[0])
plt.plot(df_predict[0])
plt.gcf().autofmt_xdate()
plt.legend(['Prediction','Time Series'],fontsize = 21)
plt.suptitle('Time-Series Prediction Entire Set of ' + id_all[i_id], fontsize = 23)
plt.xticks(fontsize=21 )
plt.yticks(fontsize=21 )
plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)
plt.xlabel(xlabel = 'Date',fontsize = 21)
plt.show()

# Plot test Set 
plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
plt.axvline(x=model_data.index[-28], c='r')
plt.plot(df_labels[0][-3*test_size:])
plt.plot(df_predict[0][-3*test_size:])
ax = plt.gca() 
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d')) 
plt.xticks(pd.date_range(df_labels.index[-3*test_size], df_labels.index[-1],freq='7d'))
plt.gcf().autofmt_xdate()
plt.legend(['Prediction','Time Series'],fontsize = 21)
plt.suptitle('Time-Series Prediction test Set of ' + id_all[i_id],fontsize = 23)
plt.xticks(fontsize=21 )
plt.yticks(fontsize=21 )
plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)
plt.xlabel(xlabel = 'Date',fontsize = 21)
plt.show()

## insample RMSE
MSE(dataY_plot[:-testX.size()[0]], data_predict[:-testX.size()[0]])
## outsample RMSE
MSE(dataY_plot[-testX.size()[0]:], data_predict[-testX.size()[0]:])
## RMSSE
RMSSE(series, list(flattern(data_predict[-testX.size()[0]:])))

##############################################
## multi layer
class LSTM2(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM2, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.batch_size = 1
        self.seq_length = seq_length
        
        self.LSTM2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True,dropout = 0.25)
       
        
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
         
        
        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
       
        _, (hn, cn) = self.LSTM2(x, (h_1, c_1))
     
        #print("hidden state shpe is:",hn.size())
        y = hn.view(-1, self.hidden_size)
        
        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        #print("final state shape is:",final_state.shape)
        out = self.fc(final_state)
        #out = self.dropout(out)
        #print(out.size())
        return out

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
#####  Parameters  ######################
num_epochs = 101
learning_rate = 1e-3
input_size = 1
hidden_size = 256
num_layers = 2
num_classes = 1

#####Init the Model #######################
lstm = LSTM2(num_classes, input_size, hidden_size, num_layers)
lstm.to(device)
lstm.apply(init_weights)

##### Set Criterion Optimzer and scheduler ####################
criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=100, factor =0.5 ,min_lr=1e-7, eps=1e-08)

# Train the model
for epoch in progress_bar(range(num_epochs)): 
    lstm.train()
    outputs = lstm(trainX.to(device))
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, trainY.to(device))
    loss.backward()
    
    scheduler.step(loss)
    optimizer.step()
    lstm.eval()
    valid = lstm(testX.to(device))
    vall_loss = criterion(valid, testY.to(device))
    scheduler.step(vall_loss)
    
    if epoch % 25 == 0:
      print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(),vall_loss.cpu().item()))        
    
    ## deal with CUDA out of memory problem
    outputs = outputs.cpu()
    loss = loss.cpu()
    valid = valid.cpu()
    vall_loss = vall_loss.cpu()
    torch.cuda.empty_cache()
    
######Prediction###############
lstm.eval()
train_predict = lstm(dataX.to(device))
data_predict = train_predict.cpu().data.numpy()
dataY_plot = dataY.data.numpy()

## Inverse Normalize 
data_predict = scaler.inverse_transform(data_predict)
dataY_plot = scaler.inverse_transform(dataY_plot)

## Add dates
df_predict = pd.DataFrame(data_predict)
df_predict = df_predict.set_index([model_data.index[(1+seq_length):]])
df_labels = pd.DataFrame(dataY_plot)
df_labels = df_labels.set_index([model_data.index[(1+seq_length):]])

# Plot Entire Set 
plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
plt.axvline(x=model_data.index[-28], c='r')
plt.plot(df_labels[0])
plt.plot(df_predict[0])
plt.gcf().autofmt_xdate()
plt.legend(['Prediction','Time Series'],fontsize = 21)
plt.suptitle('Time-Series Prediction Entire Set of ' + id_all[i_id], fontsize = 23)
plt.xticks(fontsize=21 )
plt.yticks(fontsize=21 )
plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)
plt.xlabel(xlabel = 'Date',fontsize = 21)
plt.show()

# Plot test Set 
plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
plt.axvline(x=model_data.index[-28], c='r')
plt.plot(df_labels[0][-3*test_size:])
plt.plot(df_predict[0][-3*test_size:])
ax = plt.gca() 
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d')) 
plt.xticks(pd.date_range(df_labels.index[-3*test_size], df_labels.index[-1],freq='7d'))
plt.gcf().autofmt_xdate()
plt.legend(['Prediction','Time Series'],fontsize = 21)
plt.suptitle('Time-Series Prediction test Set of ' + id_all[i_id],fontsize = 23)
plt.xticks(fontsize=21 )
plt.yticks(fontsize=21 )
plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)
plt.xlabel(xlabel = 'Date',fontsize = 21)
plt.show()

## insample RMSE
MSE(dataY_plot[:-testX.size()[0]], data_predict[:-testX.size()[0]])
## outsample RMSE
MSE(dataY_plot[-testX.size()[0]:], data_predict[-testX.size()[0]:])
## RMSSE
RMSSE(series, list(flattern(data_predict[-testX.size()[0]:])))

###############################################################################
## loop for all items

## define arima model predict function
def arima_pred(actual, pred_num):
    '''
    --------
    Description:
    actual is the true value on both train and test data
    pred_num is the length of test data
    --------
    Example:
    fit, pred = arima_pred(series, 28)
    '''
    ## data split
    train, test = model_selection.train_test_split(actual, test_size=pred_num)
    ## train model
    arima_model = pm.auto_arima(train, trace=False, stepwise=True, suppress_warnings=True, error_action='ignore')
    ## predict
    pred = arima_model.predict(n_periods=pred_num)
    return([arima_model.predict_in_sample(), pred])

## define single layer LSTM model predict function
def singleLSTM_pred(actual, pred_num):
    '''
    --------
    Description:
    actual is the true value on both train and test data
    pred_num is the length of test data
    --------
    Example:
    fit, pred = singleLSTM_pred(np.array(series), 28)
    '''
    
    ## data normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(actual.reshape(-1, 1))
    
    ## set seq_length to look back
    seq_length = 28
    x, y = sliding_windows(train_data_normalized, seq_length)
    
    ## define test size
    test_size = pred_num
    train_size = len(y) - 2 * test_size
    
    dataX = Variable(torch.Tensor(np.array(x)))
#    dataY = Variable(torch.Tensor(np.array(y)))
    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    validX = Variable(torch.Tensor(np.array(x[train_size:(train_size+test_size)])))
    validY = Variable(torch.Tensor(np.array(y[train_size:(train_size+test_size)])))
#    testX = Variable(torch.Tensor(np.array(x[(train_size+test_size):len(x)])))
#    testY = Variable(torch.Tensor(np.array(y[(train_size+test_size):len(y)])))
        
    #####  Parameters  ######################
    num_epochs = 51
    learning_rate = 1e-3
    input_size = 1
    hidden_size = 256
    num_layers = 1
    num_classes = 1
    
    #####Init the Model #######################
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    lstm.to(device)
    
    ##### Set Criterion Optimzer and scheduler ####################
    criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500, factor =0.5 ,min_lr=1e-7, eps=1e-08)
    
    # Train the model
    for epoch in progress_bar(range(num_epochs)): 
        lstm.train()
        outputs = lstm(trainX.to(device))
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY.to(device))
        loss.backward()
        optimizer.step()
        
        #Evaluate on validate     
        lstm.eval()
        valid = lstm(validX.to(device))
        vall_loss = criterion(valid, validY.to(device))
        scheduler.step(vall_loss)
        
#        if epoch % 25 == 0:
#          print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(), vall_loss.cpu().item()))
        
        ## deal with CUDA out of memory problem
        outputs = outputs.cpu()
        loss = loss.cpu()
        valid = valid.cpu()
        vall_loss = vall_loss.cpu()
        torch.cuda.empty_cache()
        
    ######Prediction###############
    lstm.eval()
    train_predict = lstm(dataX.to(device))
    data_predict = train_predict.cpu().data.numpy()
#    dataY_plot = dataY.data.numpy()
    
    ## Inverse Normalize 
    data_predict = scaler.inverse_transform(data_predict)
#    dataY_plot = scaler.inverse_transform(dataY_plot)

#    return([data_predict[:-test_size], data_predict[-test_size:]])
    return([list(flattern(data_predict[:-test_size])), list(flattern(data_predict[-test_size:]))])

## define single layer LSTM model predict function
def multiLSTM_pred(actual, pred_num):
    '''
    --------
    Description:
    actual is the true value on both train and test data
    pred_num is the length of test data
    --------
    Example:
    fit, pred = singleLSTM_pred(np.array(series), 28)
    '''
    
    ## data normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(actual.reshape(-1, 1))
    
    ## set seq_length to look back
    seq_length = 28
    x, y = sliding_windows(train_data_normalized, seq_length)
    
    ## define test size
    test_size = pred_num
    train_size = len(y) - 2 * test_size
    
    dataX = Variable(torch.Tensor(np.array(x)))
#    dataY = Variable(torch.Tensor(np.array(y)))
    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    validX = Variable(torch.Tensor(np.array(x[train_size:(train_size+test_size)])))
    validY = Variable(torch.Tensor(np.array(y[train_size:(train_size+test_size)])))
#    testX = Variable(torch.Tensor(np.array(x[(train_size+test_size):len(x)])))
#    testY = Variable(torch.Tensor(np.array(y[(train_size+test_size):len(y)])))
        
    #####  Parameters  ######################
    num_epochs = 26
    learning_rate = 1e-3
    input_size = 1
    hidden_size = 256
    num_layers = 2
    num_classes = 1
    
    #####Init the Model #######################     
    lstm = LSTM2(num_classes, input_size, hidden_size, num_layers)
    lstm.to(device)
    lstm.apply(init_weights)
    
    ##### Set Criterion Optimzer and scheduler ####################
    criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=100, factor =0.5 ,min_lr=1e-7, eps=1e-08)
    
    # Train the model
    for epoch in progress_bar(range(num_epochs)): 
        lstm.train()
        outputs = lstm(trainX.to(device))
        optimizer.zero_grad()
    
        # obtain the loss function
        loss = criterion(outputs, trainY.to(device))
        loss.backward()
        
        scheduler.step(loss)
        optimizer.step()
        lstm.eval()
        valid = lstm(validX.to(device))
        vall_loss = criterion(valid, validY.to(device))
        scheduler.step(vall_loss)
        
#        if epoch % 25 == 0:
#          print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(),vall_loss.cpu().item()))        
        
        ## deal with CUDA out of memory problem
        outputs = outputs.cpu()
        loss = loss.cpu()
        valid = valid.cpu()
        vall_loss = vall_loss.cpu()
        torch.cuda.empty_cache()    
        
    ######Prediction###############
    lstm.eval()
    train_predict = lstm(dataX.to(device))
    data_predict = train_predict.cpu().data.numpy()
#    dataY_plot = dataY.data.numpy()
    
    ## Inverse Normalize 
    data_predict = scaler.inverse_transform(data_predict)
#    dataY_plot = scaler.inverse_transform(dataY_plot)

#    return([data_predict[:-test_size], data_predict[-test_size:]])
    return([list(flattern(data_predict[:-test_size])), list(flattern(data_predict[-test_size:]))])
    
###############################################################################
evaluation_metrics = pd.DataFrame()
pred_num = 28
pred_result = pd.DataFrame(columns=['id'] + ['F'+ str(i) for i in (np.arange(pred_num)+1)])
#for i_id in range(len(id_all)):
for i_id in range(1001):
    print(i_id)
    t=time.time()
        
    state_id = id_all[i_id][-4:-2]
    store_id = id_all[i_id][-4:]
    item_id = id_all[i_id][:-5]
    dept_id = item_id[:-4]
    cat_id = dept_id [:-2]
    
    ## filter for different state or item
    state = ['CA', 'TX', 'WI']
    state.remove(state_id)
    snap_del = list(map(lambda x: 'snap_' + x, state))
    
    ## merge data
    model_data = pd.merge(DF_Sales[id_all[i_id]].reset_index(), DF_calendar, on='date', how='left')  
    model_data = pd.merge(model_data, DF_sell_prices.loc[DF_sell_prices.id==id_all[i_id]], on='wm_yr_wk', how='left')
    model_data = model_data.set_index(model_data.date)
    model_data.drop(['date', 'wm_yr_wk', 'store_id', 'item_id', 'id'] + snap_del, axis=1, inplace=True)
    ## target series
    actual = np.array(model_data[id_all[i_id]])
    
    ## modeling
    ## model 1: ARIMA model
#    fit, pred = arima_pred(actual, pred_num)
    
    ## model 2: single LSTM model
#    fit, pred = singleLSTM_pred(actual, pred_num)
    
    ## model 3: multi LSTM model
    fit, pred = multiLSTM_pred(actual, pred_num)
    
    ## collect predict result
    pred_tmp = [id_all[i_id]]
    pred_tmp.extend(list(pred))
    pred_result.loc[len(pred_result)] = pred_tmp
    
    ## model evaluation    
    MSE_insample = MSE(actual[(-pred_num-len(fit)):-pred_num], fit)
    MSE_outsample = MSE(actual[-pred_num:], pred)
    RMSSE_value = RMSSE(actual, pred)
    evaluation_metrics = evaluation_metrics.append(pd.DataFrame({'id':id_all[i_id],
                                                    'MSE_insample': [MSE_insample],
                                                    'MSE_outsample': [MSE_outsample], 
                                                    'RMSSE_value': [RMSSE_value]}), ignore_index=True)
    
    print("......Calculation time: " + str(int(time.time()-t)) + " sec")
 
###############################################################################   
## output result
pred_result.to_csv(dataPath+'output/pred_result_multilstm.csv', index=False)
evaluation_metrics.to_csv(dataPath+'output/evaluation_metrics_multilstm.csv', index=False)


    