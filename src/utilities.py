import pandas as pd

def clear_csv(filename):
    open(filename, 'w').close()

def write_to_csv(data, filename):
    data_for_df = []
    for layer, layer_data in data.items():
        layer_dict = {
            'Layer': layer,
            'Attention mechanism': layer_data['Attention mechanism'],
            'Intermediate residual stream': layer_data['Intermediate residual stream'],
            'MLP output': layer_data['MLP output'],
            'Block output': layer_data['Block output']
        }

        data_for_df.append(layer_dict)
    df = pd.DataFrame(data_for_df)
    print(df)
    df.set_index('Layer', inplace=True)  # Set the 'Layer' column as the index
    df.to_csv(filename, mode='a', index=True)