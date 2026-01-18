# Wykonuje wstępną analizę danych (EDA) dla punktu 4:
# statystyki opisowe, brakujące wartości oraz korelacje

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

from src.config import (DATA_PATH, TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
from src.data_loads import load_data

from pathlib import Path
from dataclasses import dataclass
