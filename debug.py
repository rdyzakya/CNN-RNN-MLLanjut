from models import *
from layers import *

import numpy as np
import pandas as pd

def debug1():
    model = Sequential()
    dense1 = Dense(units=32, input_shape=(20,), activation='relu')
    model.add(dense1)
    dense2 = Dense(units=32, activation='relu')
    model.add(dense2)
    sigmoid = Dense(units=1, activation='sigmoid')
    model.add(sigmoid)

    data = np.random.random((10, 20))

    print(model.predict(data))

def debug2():
    lstm = LSTM(units=5, input_shape=(5, 20, 10), return_sequences=False, random_seed=0)

    lstm.set_weight_and_bias((20,10))

    x = np.random.random((5,20,10))

    # print(lstm.weights)
    # print("X : ", x)

    print(lstm.forward(x))

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
        model_input = result[-window_size:].reshape(1,-1)   
        print(model_input.shape)     
        predicted = model.predict(model_input)[0]
        print(predicted)
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

def window_dataset(df,col_name,window_size,with_y=True,derivative=0):
    new_dataset = dict()
    series = df[col_name]    
    
    for i in range(derivative):
        series_t_minus_1 = series.iloc[:-1].reset_index(drop=True)
        series_t_minus_0 = series.iloc[1:].reset_index(drop=True)
        
        series = series_t_minus_0 - series_t_minus_1
    
    if window_size > series.shape[0]:
        raise Exception(f"Window size cannot be more than series size. Window size : {window_size} , Series size : {series.shape[0]}")
    
    for xi in range(window_size):
        x = f"t-{window_size-xi}"
        new_dataset[x] = list()
    if with_y:
        y = f"t-0"
        new_dataset[y] = list()
    
    last = series.shape[0]-window_size
    if not with_y:
        last += 1
    
    for i in range(0,last):
        window = series.iloc[i:i+window_size].tolist()
        for xi in range(window_size):
            x = f"t-{window_size-xi}"
            new_dataset[x].append(window[xi])
        
        if with_y:
            predicted = series.iloc[i+window_size]
            # print(predicted)
            y = f"t-0"
            new_dataset[y].append(predicted)            
    
    return pd.DataFrame(new_dataset)

def debug3():
    train_path = "datasets/ETH-USD-Train.csv"
    test_path = "datasets/ETH-USD-Test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # ["Open","Close","High","Low","Volume"]
    train_windowed = window_dataset(train,"Close",5,with_y=True,derivative=0)    
    # print(np.array(train_windowed))
    model = Sequential()
    window_size = 20
    print(len(test))
    lstm1 = LSTM(units=5, input_shape=(1, 5, 5), return_sequences=True, random_seed=0)    
    # lstm1 = LSTM(units=5, input_shape=(5, 20, 10), return_sequences=True, random_seed=0)    
    model.add(lstm1)
    lstm2 = LSTM(units=5, return_sequences=False, random_seed=0)      
    model.add(lstm2)
    dense1 = Dense(units=64, activation='linear')
    model.add(dense1)
    dense2 = Dense(units=1, activation='relu')
    model.add(dense2)

    # data = np.random.random((5,20, 10))        
    # for i in range(len(test)):
    #     model_input = result[-window_size:].reshape(1,-1)        
    #     predicted = model.predict(model_input)[0]
    #     result = np.append(result,predicted)
    predict_close = autoregressive_forecast(train,test,"Close",5,model,derivative=0)
    predict_Open = autoregressive_forecast(train,test,"Open",5,model,derivative=0)
    predict_High = autoregressive_forecast(train,test,"High",5,model,derivative=0)
    predict_Low = autoregressive_forecast(train,test,"Low",5,model,derivative=0)
    predict_Volume = autoregressive_forecast(train,test,"Volume",5,model,derivative=0)
    df = pd.DataFrame()
    df["Close"] = predict_close
    df["Open"] = predict_Open
    df["High"] = predict_High
    df["Low"] = predict_Low
    df["Volume"] = predict_Volume
    # # print(model.predict(train_windowed.values))
    # # print(model.predict(data))
    print(df)

def debug4():
    model = Sequential()
    lstm1 = LSTM(units=5, input_shape=(3, 5, 5), return_sequences=True, random_seed=0)
    model.add(lstm1)
    lstm2 = LSTM(units=5, return_sequences=False, random_seed=0)
    model.add(lstm2)
    dense = Dense(units=1, activation='sigmoid')
    model.add(dense)

    data = np.random.random((3,5, 5))
    print(model.predict(data))

if __name__ == "__main__":
    # debug1()
    # debug2()
    debug3()
    # debug4()