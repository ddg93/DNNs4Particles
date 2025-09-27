#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:07:11 2025

@author: davide
"""


import numpy as np
from sklearn.model_selection import train_test_split


def load_data(cnf):
    """
    Load angles, AABB components and vector componenets data from the particle/boxlist_file.
    Returns: X_train, X_test, y_train, y_test
    """
    a0, b0, c0, dx, dy, dz, n1, n2, n3 = np.loadtxt(cnf.data.particle+cnf.data.boxlist_file).T
    #stack the AABB and vector
    X = np.column_stack([dx, dy, dz])
    Y = np.column_stack([n1, n2, n3])
    ###apply the train test split
    return train_test_split(X, Y, test_size=cnf.training.test_size, random_state=0)
