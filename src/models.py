#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:07:11 2025

@author: davide
"""


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from tensorflow.keras.models import Model

def build_and_compile_model(cnf):
    
    #prepare the optimizer
    opt_cfg = cnf.training.optimizer
    opt_type = getattr(tf.keras.optimizers, opt_cfg.type)
    optimizer = opt_type(learning_rate=cnf.training.learning_rate)

    #build the model
    model = build_model(input_dim=cnf.model.input_dim,dense_layers=cnf.model.dense_layers,act_func=cnf.model.activation,L2_weight=cnf.model.L2_weight,
                        drop_out=cnf.model.drop_out,dropout_rate=cnf.model.dropout_rate,output_dim=cnf.model.output_dim)

    #define the custom loss function
    def custom_loss(y_true,y_pred):
        """
        Calculate the loss and the mean squared L2 difference between predicted and true particle orientation vectors
        Add a penalty for negative n3 values.
        """
        norm_loss = tf.norm((y_true-y_pred),axis=-1)
        loss = tf.reduce_mean(tf.square(norm_loss))
        ####penalty for negative n3 values
        penalty = tf.reduce_mean(tf.nn.relu(-y_pred[:,2]))    
        return loss + penalty * cnf.model.loss_weight
    
    #compile the model
    model.compile(loss=custom_loss, optimizer=optimizer)
    return model

def build_model(input_dim, dense_layers=[32, 32, 32], act_func="tanh",
    L2_weight=0.001,drop_out=False,dropout_rate=0.1,output_dim=3):
    """
    Build a fully connected NN with optional dropout and L2 normalization at output.
    Args:
        input_dim (int): Number of input features.
        dense_layers (list): List of hidden layer sizes.
        act_func (str): Activation function for all hidden layers.
        L2_weight (float): L2 regularization strength.
        drop_out (bool): Whether to add Dropout after each hidden layer.
        dropout_rate (float): Dropout rate (if drop_out=True).
        output_dim (int): Size of final output layer (default=3).
    Returns:
        tf.keras.Model
    """
    # Input
    inputs = Input(shape=(input_dim,))
    z = inputs
    # Hidden layers
    for units in dense_layers:
        z = Dense(units,activation=act_func,kernel_regularizer=tf.keras.regularizers.l2(L2_weight),)(z)
        if drop_out:
            z = Dropout(rate=dropout_rate)(z)
    # Final layer
    final_dense = Dense(output_dim, activation="linear")(z)
    # L2-normalized output
    outputs = Lambda(lambda n: tf.math.l2_normalize(n, axis=-1, epsilon=1e-12))(final_dense)
    # Model
    return Model(inputs=inputs, outputs=outputs)