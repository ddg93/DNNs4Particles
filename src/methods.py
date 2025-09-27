#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:07:11 2025

@author: davide
"""


import yaml

class ConfigSection:
    """Wraps a dict into an object with attribute access."""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = ConfigSection(v)
            setattr(self, k, v)
    def __repr__(self):
        return f"{self.__dict__}"


class Configuration:
    """Loads a YAML config and makes nested sections accessible as attributes."""
    def __init__(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            if isinstance(v, dict):
                v = ConfigSection(v)
            setattr(self, k, v)
    def __repr__(self):
        return f"Configuration({self.__dict__})"



def evaluate_model(model,history,X_train,X_test,Y_train,Y_test,cnf):
    """
    Evaluate the model training
    by plotting the training losses and the final epoch inference on both test and train sets
    """
    #First, plot the training and testing set losses against the model training epochs
    plt.figure(figsize=(3,2))
    plt.plot(np.arange(1,len(history.history['loss'])+1,1),history.history['loss'])
    plt.plot(np.arange(1,len(history.history['val_loss'])+1,1),history.history['val_loss'])
    plt.yscale('log');
    plt.xlabel(r'$epochs$')
    plt.ylabel(r'$Loss$'); 
    plt.savefig(cnf.data.particle+'Losses.png',format='png',dpi=300);
    
    #Second, plot the model inference accuracy on both training and testing sets
    forecast_train = model.predict(X_train)
    forecast_test = model.predict(X_test)
    fig,axs = plt.subplots(1,2,figsize=(6,2))
    axs[0].hist(np.linalg.norm(forecast_train-Y_train,axis=-1),bins=50,label=r'$Train$');
    axs[0].hist(np.linalg.norm(forecast_test-Y_test,axis=-1),bins=50,label=r'$Test$');
    axs[0].legend();
    axs[0].set_xlim(0,1)
    axs[0].set_xlabel(r'$\vert\vert \vec{n}_{true}-\vec{n}_{pred}\vert\vert_2$')
    axs[0].set_ylabel(r'$Occurences$');
    axs[1].hist(abs(forecast_train[:,0]-Y_train[:,0]),label=r'$n_1$');
    axs[1].hist(abs(forecast_train[:,1]-Y_train[:,1]),label=r'$n_2$');
    axs[1].hist(abs(forecast_train[:,2]-Y_train[:,2]),label=r'$n_3$');
    axs[1].legend();
    plt.savefig(cnf.data.particle+'Errors.png',format='png',dpi=300);

