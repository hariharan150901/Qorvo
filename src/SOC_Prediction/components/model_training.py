import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
import keras
from pathlib import Path
from SOC_Prediction.entity.config_entity import TrainModelConfig

class Training:
    def __init__(self, config: TrainModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            Path(self.config.model_path)
        )

    def train_val_test_generator(self):
        data = pd.read_csv(self.config.training_data)
        x_data = data.iloc[::self.config.sampling_rate,[1,2,3]].values
        y_data = data.iloc[::self.config.sampling_rate,[4]].values
        n_samples = len(x_data)
        
        if self.config.should_avg_be_the_feature:
            voltage = x_data[:,1]
            avg_voltage = self.rollavg_convolve_edges(voltage,self.config.avg_window).reshape(-1,1)
            current = x_data[:,0]
            avg_current = self.rollavg_convolve_edges(current,self.config.avg_window).reshape(-1,1)
            x_data = np.concatenate((avg_voltage,avg_current),axis =1)
            
        ###Write code for derivative here
        
        
        
        self.X_train = x_data[:int(self.config.n_fraction*n_samples),:]
        self.X_val = x_data[int(self.config.n_fraction*n_samples):int(0.8*n_samples),:]
        self.X_test = x_data[int(0.8*n_samples):,:]
        self.y_train = y_data[:int(self.config.n_fraction*n_samples),:]
        self.y_val = y_data[int(self.config.n_fraction*n_samples):int(0.8*n_samples),:]
        self.y_test = y_data[int(0.8*n_samples):,:]

        self.y_train = (self.y_train-np.min(self.y_train))/(np.max(self.y_train)-np.min(self.y_train))
        self.y_val = (self.y_val-np.min(self.y_val))/(np.max(self.y_val)-np.min(self.y_val))
        self.y_test = (self.y_test-np.min(self.y_test))/(np.max(self.y_test)-np.min(self.y_test))
        

        self.X_train = self.wrapper(self.X_train,self.config.window_length)
        self.X_val = self.wrapper(self.X_val,self.config.window_length)
        self.X_test = self.wrapper(self.X_test,self.config.window_length)
        self.y_train = self.y_train[:-self.config.window_length]
        self.y_val = self.y_val[:-self.config.window_length]
        self.y_test = self.y_test[:-self.config.window_length]
        
        
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_val = self.sc.transform(self.X_val)
        self.X_test = self.sc.transform(self.X_test)

        self.X_train = self.X_train.reshape(-1,self.config.window_length+1,x_data.shape[1])
        self.X_val  = self.X_val.reshape(-1,self.config.window_length+1,x_data.shape[1])
        self.X_test  = self.X_test.reshape(-1,self.config.window_length+1,x_data.shape[1])
        


    def rollavg_convolve_edges(self,a,n):
        assert n%2==1
        return scipy.signal.convolve(a,np.ones(n,dtype='float'), 'same')/scipy.signal.convolve(np.ones(len(a)),np.ones(n), 'same')  

        
        
    def wrapper(self,x_data, window):
        data_modified = x_data
        for i in range(1,window+1):
            data_modified = np.concatenate((data_modified,np.roll(x_data,i,axis=0)),axis = 1)

        return data_modified[:-window,:]    

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def train(self):
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size = self.config.batch_size,
            epochs = self.config.epochs,
            validation_data = (self.X_val,self.y_val)
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )