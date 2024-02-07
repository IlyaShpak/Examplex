import csv
import pandas as pd
data = pd.read_csv("spam.csv", encoding='latin-1')
train_data = data.iloc[0:5000]
test_data = data.iloc[5000:5572]
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)
