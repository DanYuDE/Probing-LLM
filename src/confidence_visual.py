import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

def extractWeights(data, startLine, numLayers, column_name):
    weight = {}
    for i in range(startLine, startLine + numLayers):
        layer = data.iloc[i]['Layer']
        attentionList = ast.literal_eval(data.iloc[i][column_name])
        tokens, weights = zip(*attentionList)
        weight[layer] = (tokens, weights)
    return weight

def plotAttention(attention_weights, set_num):
    num_layers = len(attention_weights)
    max_tokens = max(len(weights[0]) for weights in attention_weights.values())
    heatmap_data = np.zeros((num_layers, max_tokens))
    layer_labels = [f'Layer {i + 1}' for i in range(num_layers)]

    for i, (layer, (tokens, weights)) in enumerate(attention_weights.items()):
        for j, weight in enumerate(weights):
            heatmap_data[i, j] = weight

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')

    # Annotate each cell with the corresponding token
    for i, (layer, (tokens, weights)) in enumerate(attention_weights.items()):
        for j, token in enumerate(tokens):
            ax.text(j, i, token, ha='center', va='center', color='w')

    ax.set_xticks([])
    ax.set_yticks(np.arange(num_layers))
    ax.set_yticklabels(layer_labels)
    ax.set_title(f"Attention Probing Visualization - Set {set_num}")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Attention weight', rotation=-90, va="bottom")

    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()

def plotBlock(block_output, set_num):
    num_layers = len(block_output)
    max_tokens = max(len(weights[0]) for weights in block_output.values())
    heatmap_data = np.zeros((num_layers, max_tokens))
    layer_labels = [f'Layer {i + 1}' for i in range(num_layers)]

    for i, (layer, (tokens, weights)) in enumerate(block_output.items()):
        for j, weight in enumerate(weights):
            heatmap_data[i, j] = weight

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')

    # Annotate each cell with the corresponding token
    for i, (layer, (tokens, weights)) in enumerate(block_output.items()):
        for j, token in enumerate(tokens):
            ax.text(j, i, token, ha='center', va='center', color='w')

    ax.set_xticks([])
    ax.set_yticks(np.arange(num_layers))
    ax.set_yticklabels(layer_labels)
    ax.set_title(f"Block Output Visualization - Set {set_num}")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Block Output Weight', rotation=-90, va="bottom")

    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()


file_path = '../output/new_output3.csv'
data = pd.read_csv(file_path)

num_layers_per_set = 32
num_sets = 10

for set_num in range(num_sets):
    start_line = set_num * (num_layers_per_set + 1)  # +1 to account for the header row
    # attention_weights = extractWeights(data, start_line, num_layers_per_set, 'Attention mechanism')
    blockOutput = extractWeights(data, start_line, num_layers_per_set, 'Block output')
    # plotAttention(attention_weights, set_num + 1)
    plotBlock(blockOutput, set_num + 1)
