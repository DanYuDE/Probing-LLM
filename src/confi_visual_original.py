import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go

def extractWeights(data, startLine, numLayers, column_name):
    weight = {}
    for i in range(startLine, startLine + numLayers):
        layer = data.iloc[i]['Layer']
        t_wList = ast.literal_eval(data.iloc[i][column_name])
        tokens, weights = zip(*t_wList)
        weight[layer] = (tokens, weights)
    return weight


def create_interactive_heatmap(tokenWeights, set_num, title):
    num_layers = len(tokenWeights)
    max_tokens = max(len(weights[0]) for weights in tokenWeights.values())
    heatmap_data = np.zeros((num_layers, max_tokens))
    annotations = []
    custom_hover_data = np.empty((num_layers, max_tokens), dtype=object)

    for i, (layer, (tokens, weights)) in enumerate(tokenWeights.items()):
        for j, weight in enumerate(weights):
            heatmap_data[i, j] = weight
            custom_hover_data[i][j] = f'Token: {tokens[j]}<br>Layer: {layer}<br>Confidence: {weight:.2f}'
            annotations.append(
                dict(
                    showarrow=False,
                    x=j,
                    y=i,
                    text=tokens[j],
                    xref="x",
                    yref="y",
                    font=dict(color='white')
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f'{i}' for i in range(max_tokens)],
        y=[f'{i}' for i in range(num_layers)],
        colorscale='Viridis',
        # hoverongaps=False
        hoverongaps=False,
        # hoverinfo='none',  # Disable default hoverinfo
        text=custom_hover_data,  # Set custom hover text
        hovertemplate='%{text}'  # Use custom text for hovertemplate
    ))

    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(ticks='', side='top'),
        yaxis=dict(ticks=''),
        width=900,  # Adjust as needed
        height=900  # Adjust as needed
    )

    # fig.show()
    # plot(fig, filename='heatmap.html', auto_open=True)
    return fig

def save_html_with_two_figures(att_fig, block_fig, file_name):
    # Create an HTML string to hold both figures side by side
    html_string = f"""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div style="display: flex; justify-content: space-around;">
            <div style="width: 45%;">{att_fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            <div style="width: 45%;">{block_fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        </div>
    </body>
    </html>
    """

    # Write the HTML string to a file
    with open(file_name, 'w') as f:
        f.write(html_string)

def plotGraph(tokenWeights, set_num, ax=None, text=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 20))
    num_layers = len(tokenWeights)
    max_tokens = max(len(weights[0]) for weights in tokenWeights.values())
    heatmap_data = np.zeros((num_layers, max_tokens))
    layer_labels = [f'Layer {i + 1}' for i in range(num_layers)]

    for i, (layer, (tokens, weights)) in enumerate(tokenWeights.items()):
        for j, weight in enumerate(weights):
            heatmap_data[i, j] = weight

    # fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')

    # Annotate each cell with the corresponding token
    for i, (layer, (tokens, weights)) in enumerate(tokenWeights.items()):
        for j, token in enumerate(tokens):
            ax.text(j, i, token, ha='center', va='center', color='w')

    ax.set_xticks([])
    ax.set_yticks(np.arange(num_layers))
    ax.set_yticklabels(layer_labels)
    ax.set_title(f"{text} Visualization - Set {set_num}")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Weight', rotation=-90, va="bottom")

    plt.tight_layout()  # Adjust layout to fit everything
    # plt.show()
    return ax.figure

file_path = '../output/new_output3.csv'
data = pd.read_csv(file_path)

num_layers_per_set = 32
num_sets = 13

for set_num in range(num_sets):
    start_line = set_num * (num_layers_per_set + 1)  # +1 to account for the header row
    attention_weights = extractWeights(data, start_line, num_layers_per_set, 'Attention mechanism')
    blockOutput = extractWeights(data, start_line, num_layers_per_set, 'Block output')

    # fig_attention = plotAttention(attention_weights, set_num + 1)
    # fig_block = plotBlock(blockOutput, set_num + 1)
    # plotAttention(attention_weights, set_num + 1)
    # plotBlock(blockOutput, set_num + 1)

    # fig, axs = plt.subplots(1, 2, figsize=(40, 20))  # Adjust figsize as needed
    # plotGraph(attention_weights, set_num + 1, axs[0], "Attention Probing")
    # plotGraph(blockOutput, set_num + 1, axs[1], "Block Output")
    #
    # plt.tight_layout()
    # plt.savefig(f'../image/heatmap_set_{set_num + 1}.png', dpi=300)  # Save each set as a high-resolution image
    # plt.show()

    att_fig = create_interactive_heatmap(attention_weights, start_line, f"Attention Probing Visualization - Set {set_num + 1}")
    block_fig = create_interactive_heatmap(blockOutput, start_line, f"Block Output Visualization - Set {set_num + 1}")

    # Generate the filename based on the set number
    filename = f'../html/heatmap_set_{set_num + 1}.html'

    # Save the figures to an HTML file
    save_html_with_two_figures(att_fig, block_fig, filename)
