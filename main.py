from models import *
from layers import *

import random
import numpy as np
import pandas as pd

def create_model(random_seed_1:int,random_seed_2:int):
    model = Sequential()
        
    lstm1 = LSTM(units=5, input_shape=(1, 5, 1), return_sequences=True, random_seed=random_seed_1)    
    
    model.add(lstm1)
    lstm2 = LSTM(units=5, return_sequences=False, random_seed=random_seed_2)      
    model.add(lstm2)
    dense1 = Dense(units=64, activation='linear')
    model.add(dense1)
    dense2 = Dense(units=1, activation='relu')
    model.add(dense2)

    return model

def autoregressive_forecast(train,test,colname,window_size,model,derivative=0):
    series = train[colname]
    series_derivative = {0 : series.copy()}
    for i in range(1,derivative+1):
        series_t_minus_1 = series.iloc[:-1].reset_index(drop=True)
        series_t_minus_0 = series.iloc[1:].reset_index(drop=True)
        
        series = series_t_minus_0 - series_t_minus_1
        
        series_derivative[i] = series.copy()
    
    if window_size > series.shape[0]:
         raise Exception(f"Window size cannot be more than series size. Window size : {window_size} , Series size : {series.shape[0]}")
    
    result = series.tail(window_size).values # .reshape(1,-1)        
    for i in range(len(test)):
        # print(result[-window_size:].reshape(1,-1,1).shape)
        # exit()
        model_input = result[-window_size:].reshape(1,-1,1)   
        predicted = model.predict(model_input)[0]
        # print(predicted)
        result = np.append(result,predicted)    
    result = result[window_size:]        
    result_derivative = {}
    result_derivative[derivative] = result.tolist()
    
    for i in reversed(range(1,derivative+1)):
        c_series = series_derivative[i-1]
        last_value = c_series.iloc[-1]
        result_derivative[i-1] = list()
        for j in range(len(test)):
            last_value += result_derivative[i][j]
            result_derivative[i-1].append(last_value)    
    return result_derivative[0]

if __name__ == "__main__":
    train_path = "datasets/ETH-USD-Train.csv"
    test_path = "datasets/ETH-USD-Test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    columns = ["Open","Close","High","Low","Volume"]
    # columns = ["Close","Open"]
    model = create_model(random_seed_2=random.randint(1,2000),random_seed_1=random.randint(1,2000))
    df = pd.DataFrame()
    for i in columns:        
        model_copy = model
        df[i] = autoregressive_forecast(train,test,i,5,model_copy,derivative=0)    
    
    print(df)
