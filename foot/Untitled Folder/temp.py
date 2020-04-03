import pandas as pd

# lr_df = pd.read_csv('lr_foot.csv').sort_values(by="filename")

lr_df = pd.read_csv('../newest.csv')
print(lr_df.tail())
print(lr_df.shape)
# print(lr_df[lr_df.filename == '13457.jpg'])

