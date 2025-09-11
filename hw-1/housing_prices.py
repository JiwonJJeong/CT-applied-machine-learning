# https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/
import pandas as pd
from matplotlib import pyplot as plt

file_path = "./data/train.csv"

def dist_1st_floor_sqft():
    attributes = pd.read_csv(file_path, usecols=[0,1,2,3], header=None, names=['sepal length in cm', 'sepal width in cm','petal length in cm', 'petal width in cm'])