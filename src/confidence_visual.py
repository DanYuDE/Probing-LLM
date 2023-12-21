import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
import dash
from dash import Dash, html, dcc, Input, Output, callback, State
from plotly.graph_objs import Figure

# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

file_path = '../output/test.json'
# data = pd.read_csv(file_path)
data = pd.read_json(file_path)
# print(len(data))
num_layers_per_set = len(data)
num_sets = int(len(data)/num_layers_per_set)
print(num_sets)
# Layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='set-selector',
        options=[{'label': f'Set {i + 1}', 'value': i} for i in range(num_sets)],
        value=0
    ),
    html.Div(id='graphs-container')
])


def extractWeightsCSV(data, startLine, numLayers, column_name):
    weight = {}
    for i in range(startLine, startLine + numLayers):
        layer = data.iloc[i]['Layer']
        t_wList = ast.literal_eval(data.iloc[i][column_name])
        tokens, weights = zip(*t_wList)
        weight[layer] = (tokens, weights)
    return weight

def extractWeightsJSON(data, startLine, numLayers, column_name):
    weight = {}
    for i in range(startLine, startLine + numLayers):
        layer = data.iloc[i]['Layer']
        t_wList = data.iloc[i][column_name]
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


# Callback to update graphs based on selected set
@app.callback(
    Output('graphs-container', 'children'),
    Input('set-selector', 'value'))
def update_graphs(set_num):
    start_line = set_num * (num_layers_per_set + 1)  # +1 for header row
    attention_weights = extractWeightsJSON(data, start_line, num_layers_per_set, 'Attention mechanism')
    blockOutput = extractWeightsJSON(data, start_line, num_layers_per_set, 'Block output')

    att_fig = create_interactive_heatmap(attention_weights, set_num,
                                         f"Attention Probing Visualization - Set {set_num + 1}")
    block_fig = create_interactive_heatmap(blockOutput, set_num, f"Block Output Visualization - Set {set_num + 1}")

    return html.Div([
        html.Div(dcc.Graph(figure=att_fig, id='att-heatmap'), style={'width': '50%', 'display': 'inline-block'}),
        html.Div(dcc.Graph(figure=block_fig, id='block-heatmap'), style={'width': '50%', 'display': 'inline-block'})
    ], style={'display': 'flex'})


# Utility function to construct hover text for a given token
def construct_hover_text(tokenWeights, hovered_token):
    hover_texts = []
    for layer, (tokens, weights) in tokenWeights.items():
        layer_hover_texts = []
        for token, weight in zip(tokens, weights):
            if token == hovered_token:
                layer_hover_texts.append(f"{token}<br>Layer: {layer}<br>Confidence: {weight} (Corresponding token)")
            else:
                layer_hover_texts.append(f"{token}<br>Layer: {layer}<br>Confidence: {weight}")
        hover_texts.append(layer_hover_texts)
    return hover_texts


# Callback to synchronize hover data between the two graphs
@app.callback(
    Output('att-heatmap', 'figure'),
    Output('block-heatmap', 'figure'),
    Input('att-heatmap', 'hoverData'),
    Input('block-heatmap', 'hoverData'),
    State('set-selector', 'value'),
    State('att-heatmap', 'figure'),
    State('block-heatmap', 'figure'))
def synchronize_hover(att_hoverData, block_hoverData, set_num, att_fig_dict, block_fig_dict):
    ctx = dash.callback_context
    if not ctx.triggered:
        return att_fig_dict, block_fig_dict

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Convert dictionary to Plotly Figure objects
    att_fig = go.Figure(att_fig_dict)
    block_fig = go.Figure(block_fig_dict)

    try:
        if trigger_id == 'att-heatmap' and att_hoverData:
            hovered_token = att_hoverData['points'][0]['text'].split('<br>')[0].split(': ')[1]
            print("hovered token = ", hovered_token)
            hovered_layer = int(att_hoverData['points'][0]['y'])
            hovered_index = int(att_hoverData['points'][0]['x'])
            print(hovered_layer)
            print(hovered_index)
            # Update hovertext for the block heatmap based on the hovered token
            # block_weights = extractWeights(data, set_num * (num_layers_per_set + 1), num_layers_per_set, 'Block output')
            # print("block weights = ", block_weights)
            # block_hover_texts = construct_hover_text(block_weights, hovered_token)
            # print("block hover texts = ", block_hover_texts)
            for trace in block_fig['data']:
                if trace['type'] == 'heatmap':
                    trace['hovertext'][hovered_layer][hovered_index] += ' (Corresponding token)'

        elif trigger_id == 'block-heatmap' and block_hoverData:
            hovered_token = block_hoverData['points'][0]['text'].split('<br>')[0].split(': ')[1]
            hovered_layer = int(block_hoverData['points'][0]['y'])
            hovered_index = int(block_hoverData['points'][0]['x'])
            print(hovered_token)
            print(hovered_layer)
            print(hovered_index)
            # Update hovertext for the attention heatmap based on the hovered token
            # att_weights = extractWeights(data, set_num * (num_layers_per_set + 1), num_layers_per_set,
            #                              'Attention mechanism')
            # print("att weights = ", att_weights)
            # att_hover_texts = construct_hover_text(att_weights, hovered_token)
            # print("att hover texts = ", att_hover_texts)
            for trace in att_fig['data']:
                if trace['type'] == 'heatmap':
                    trace['hovertext'][hovered_layer][hovered_index] += ' (Corresponding token)'

    except Exception as e:
        print("Error during hover synchronization:", e)

    return att_fig.to_dict(), block_fig.to_dict()


def update_hover_info_for_other_graph(fig_dict, token):
    # Check if data exists in the figure dictionary
    if 'data' not in fig_dict:
        print("No data key in figure dictionary.")
        return

    # Iterate over each trace in the figure data
    for trace in fig_dict['data']:
        # Check if the trace type is 'heatmap'
        if 'type' in trace and trace['type'] == 'heatmap':
            # Check if hovertext is present
            if 'hovertext' not in trace or not isinstance(trace['hovertext'], list):
                print("No hovertext found or hovertext is not a list.")
                continue

            # Update hovertext
            new_hovertext = []
            for row in trace['hovertext']:
                new_row = []
                if not isinstance(row, list):
                    print("Row in hovertext is not a list.")
                    continue
                for cell_text in row:
                    if f'Token: {token}' in cell_text:
                        new_row.append(cell_text + ' (Corresponding token)')
                    else:
                        new_row.append(cell_text)
                new_hovertext.append(new_row)
            trace['hovertext'] = new_hovertext


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

