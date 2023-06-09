import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('./subtracted_data.csv')


scaler = MinMaxScaler()
numeric_columns = [
    'total_time',
    'prepad',
    'spec',
    'mag',
    'before mask',
    'x device change',
    'mask',
    'ispec',
    'x device change 2',
    'xt + x'
]

df_normalized = df.copy()

# Normalize the columns
df_normalized[numeric_columns] = pd.DataFrame(scaler.fit_transform(df_normalized[numeric_columns]), columns=numeric_columns)


df_normalized = df_normalized.set_index(['model_name', 'grad'])

# Create a heatmap
sns.heatmap(df_normalized[numeric_columns], annot=df[numeric_columns], cmap='coolwarm')

# Display the plot
plt.show()