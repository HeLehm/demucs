import pandas as pd

# Read the CSS file into a pandas DataFrame
df = pd.read_csv('./demucs_performance.csv')

df = df.groupby(['model_name', 'grad'])

new_data = []

cols_to_subtract = [
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

for group_name, group_data in df:
    # subtract row where device = cpu from row where device = mps
    # and save the result in a new DataFrame
    new_df = group_data[group_data['device'] == 'mps'][cols_to_subtract].reset_index(drop=True) - group_data[group_data['device'] == 'cpu'][cols_to_subtract].reset_index(drop=True)
    # add the model name and grad to the new DataFrame
    new_df['model_name'] = group_name[0]
    new_df['grad'] = group_name[1]
    # madke model_name and grad the first two columns
    new_df = new_df[['model_name', 'grad', *cols_to_subtract]]
    # append the new DataFrame to a list
    new_data.append(new_df)

# Concatenate all the DataFrames in the list into a single DataFrame
subtracted_df = pd.concat(new_data)


# Save the subtracted DataFrame to a new CSV file
subtracted_df.to_csv('./subtracted_data.csv', index=False)
