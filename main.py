#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:07:11 2025
@author: davide
"""

from src.methods import Configuration, evaluate_model
from src.models import build_model
from src.data_handler import load_data
import 

def main(cnf):
    """
    Train a neural network to regress the particle orientation vector
    from its Axes-Aligned Bounding Boxes, on synthetic data. 
    Hyper-parameters passed in the cnf Configuration instance.
    """
    #load the data and split in train and test X and Y sets
    X_train, X_test, Y_train, Y_test = load_data(cnf)

    #build and compile the model
    model = build_and_compile_model(cnf)

    #train the model
    history = model.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=cnf.training.epochs, batch_size=cnf.training.batch_size, verbose=1)
    
    #plot the model training evaluation
    evaluate_model(model,history,X_train,X_test,Y_train,Y_test,cnf)

    #save the model
    if cnf.data.model_saving:
        model.save(cnf.data.particle+cnf.data.model_saving)


if __name__ == '__main__':
    cnf = Configuration('configuration.yaml')
    main(cnf)
