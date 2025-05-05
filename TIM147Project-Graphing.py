import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV file
dtype_dict = {
    'column_name_1': str,
    'column_name_2': float,
    # etc.
}

df = pd.read_csv('/Users/rithviks/Desktop/TIM147/FPA_FOD_Plus.csv', dtype=dtype_dict)

# Preview the data
print(df.head())

#All cols:
all_cols = df.columns
print(all_cols)

#numerical_cols = {
#        }

# Histogram for each numerical column
#for col in numerical_cols:
#    plt.figure(figsize=(8, 5))
#    sns.histplot(df[col], kde=True, bins=30)
#    plt.title(f'Distribution of {col}')
#    plt.xlabel(col)
#    plt.ylabel('Frequency')
#    plt.show()
