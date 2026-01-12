# Definiuje i trenuje modele uczenia maszynowego, w tym model bazowy oraz alternatywne klasyfikatory


import numpy as np 

# base model
from sklearn.linear_model import LinearRegression

# tuning
from sklearn.model_selection import GridSearchCV

# models learning
from sklearn.model_selection import train_test_split

# alternatywne klasyfikatory
from sklearn.ensemble import RandomForestClassifier

# alternatywne klasyfikatory
from sklearn.svm import SVC