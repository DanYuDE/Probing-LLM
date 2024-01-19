import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
import dash
from dash import Dash, html, dcc, Input, Output, callback, State, ClientsideFunction
from plotly.graph_objs import Figure

# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# file_path = '../output/llama2_7b_chat_hf_res.csv'
file_path = '../output/llama2_7b_hf_res.csv'
data = pd.read_csv(file_path)

# Read the text file
with open('../files/outtest.txt', 'r') as file:
    sentences = file.read().strip().split('\n')  # Assuming each sentence is on a new line

sampleText = """<<SYS>>
Role and goal: 
Your role is to act as an intelligent problem-solver, tasked with selecting the correct answer from a set of multiple-choice options. Your goal is to carefully analyze the question and each of the provided options, applying your extensive knowledge base and reasoning skills to determine the most accurate and appropriate answer.

Context:
The input text is a question with multiple-choice options. The correct answer is indicated by the option label A, B, C, or D.
1. Question: A clear query requiring a specific answer.
2. Options: A list of possible answers labeled with possible answer A.First possible answer B.Second possible answer C.Third possible answer D.Fourth possible answer

Instructions:
Analyze the question and options provided.
Use your knowledge to assess each option.
Employ reasoning to eliminate clearly incorrect options.
Identify the most accurate answer based on the information given.
Conclude by justifying your selection, clearly indicating your choice by referencing the option label A, B, C, or D.
You should only output one capitalized letter indicating the correct answer.

Example:
Input: // you will receive the question and options here.
Output: The correct answer is {one of A, B, C, D} // you will output the correct answer, replace {one of A, B, C, D} with the correct option label A, B, C, or D.

Now you can start to answer the question with given options to give the correct answer.
<</SYS>>

[INST] Input: {{inputText}}[/INST]
Output: The correct answer is """

num_layers_per_set = 32
num_sets = 12
print(num_sets)
# Layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='set-selector',
        options=[{'label': f'Set {i + 1}', 'value': i} for i in range(num_sets)],
        value=0
    ),
    html.Div(id='graphs-container'),  # Graphs will be populated by a callback
    # html.Div(  # This is the new container for hover info boxes
    #     [
    #         html.Div(id='att-hover-info', style={'width': '50%', 'display': 'inline-block'}),
    #         html.Div(id='block-hover-info', style={'width': '50%', 'display': 'inline-block'})
    #     ],
    #     style={'display': 'flex'}
    # )
])


def extractWeightsCSV(data, startLine, numLayers, column_name):
    weight = {}
    for i in range(startLine, startLine + numLayers):
        layer = data.iloc[i]['Layer']
        t_wList = ast.literal_eval(data.iloc[i][column_name])
        tokens, weights = zip(*t_wList)
        weight[layer] = (tokens, weights)
    return weight


def create_interactive_heatmap(tokenWeights, title):
    num_layers = len(tokenWeights)
    max_tokens = max(len(weights[0]) for weights in tokenWeights.values())
    heatmap_data = np.zeros((num_layers, max_tokens))
    annotations = []
    hovertext = []  # Initialize an empty list for hovertext

    for i, (layer, (tokens, weights)) in enumerate(tokenWeights.items()):
        hovertext_row = []  # Initialize an empty list for this row
        for j, weight in enumerate(weights):
            heatmap_data[i, j] = weight
            hovertext_cell = f'Token: {tokens[j]}<br>Layer: {layer}<br>Confidence: {weight:.2f}'
            hovertext_row.append(hovertext_cell)  # Add cell hovertext to the row
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
        hovertext.append(hovertext_row)  # Add the completed row to hovertext

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f'{i}' for i in range(max_tokens)],
        y=[f'{i}' for i in range(num_layers)],
        colorscale='Viridis',
        hoverongaps=False,
        hoverinfo='text',
        hovertext=hovertext  # Make sure this is a list of lists.
    ))

    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(ticks='', side='top'),
        yaxis=dict(ticks=''),
        width=900,
        height=900
    )

    return fig


# Callback to update graphs based on selected set
@app.callback(
    Output('graphs-container', 'children'),
    Input('set-selector', 'value'))
def update_graphs(set_num):
    print("update_graphs called with set_num:", set_num)  # 添加的打印语句
    start_line = set_num * (num_layers_per_set + 1)  # +1 for header row
    attention_weights = extractWeightsCSV(data, start_line, num_layers_per_set, 'Attention mechanism')
    blockOutput = extractWeightsCSV(data, start_line, num_layers_per_set, 'Block output')

    att_fig = create_interactive_heatmap(attention_weights,
                                         f"Attention Probing Visualization - Set {set_num + 1}")
    block_fig = create_interactive_heatmap(blockOutput, f"Block Output Visualization - Set {set_num + 1}")

    sentence = sampleText.replace("{{inputText}}", sentences[set_num]) if set_num < len(sentences) else "No sentence available."

    return html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
        # Div for the graphs
        html.Div(style={'display': 'flex'}, children=[
            # Left column for Attention Heatmap and its hover info
            html.Div(style={'width': '50%', 'marginBottom': '20px'}, children=[
                dcc.Graph(figure=att_fig, id='att-heatmap'),
                html.Div(id='att-hover-info', style={'textAlign': 'center', 'marginTop': '10px'})
                # Adjust space above hover info
            ]),
            # Right column for Block Output Heatmap and its hover info
            html.Div(style={'width': '50%', 'marginBottom': '20px'}, children=[
                dcc.Graph(figure=block_fig, id='block-heatmap'),
                html.Div(id='block-hover-info', style={'textAlign': 'center', 'marginTop': '10px'})
                # Adjust space above hover info
            ])
        ]),
        # Separate Div for the sentence
        html.Div(sentence, style={'width': '100%', 'textAlign': 'left', 'marginTop': '20px'})
        # Adjust space above the sentence
    ])


# app.clientside_callback(
#     ClientsideFunction(namespace='clientside', function_name='synchronizeHover'),
#     [Output('att-heatmap-hoverinfo', 'children'),
#      Output('block-heatmap-hoverinfo', 'children')],
#     [Input('att-heatmap', 'hoverData'),
#      Input('block-heatmap', 'hoverData')]
# )

@app.callback(
    [Output('att-hover-info', 'children'),
     Output('block-hover-info', 'children')],
    [Input('att-heatmap', 'hoverData'),
     Input('block-heatmap', 'hoverData')],
    [State('att-heatmap', 'figure'),
     State('block-heatmap', 'figure'),
     State('att-hover-info', 'children'),
     State('block-hover-info', 'children')]
)
def update_hover_info(att_hoverData, block_hoverData, att_fig_dict, block_fig_dict, att_info_text, block_info_text):
    ctx = dash.callback_context
    if not ctx.triggered:
        # 如果没有触发的输入，则不更新
        return dash.no_update

    # 检查哪个输入被触发
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 初始化信息变量
    att_info = att_info_text if att_info_text is not None else "Hover on Attention Heatmap for info."
    block_info = block_info_text if block_info_text is not None else "Hover on Block Output Heatmap for info."

    # att_info = "Hover on Attention Heatmap for info."
    # block_info = "Hover on Block Output Heatmap for info."

    if trigger_id == 'att-heatmap' and att_hoverData:
        # hovered_token = att_hoverData['points'][0]['hovertext'].split('<br>')[0].split(': ')[1]
        hovered_layer = int(att_hoverData['points'][0]['y'])
        hovered_index = int(att_hoverData['points'][0]['x'])
        att_hovertext = att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
        att_info = f"Hovered on Attention Heatmap: {att_hovertext.replace('<br>', ', ')}"
        block_hovertext = block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
        block_info = f"Hovered on Block Output Heatmap: {block_hovertext.replace('<br>', ', ')}"


    elif trigger_id == 'block-heatmap' and block_hoverData:
        # hovered_token = block_hoverData['points'][0]['hovertext'].split('<br>')[0].split(': ')[1]
        hovered_layer = int(block_hoverData['points'][0]['y'])
        hovered_index = int(block_hoverData['points'][0]['x'])
        block_hovertext = block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
        block_info = f"Hovered on Block Output Heatmap: {block_hovertext.replace('<br>', ', ')}"
        att_hovertext = att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
        att_info = f"Hovered on Attention Heatmap: {att_hovertext.replace('<br>', ', ')}"

    return att_info, block_info


# Callback to synchronize hover data between the two graphs
# @app.callback(
#     Output('att-heatmap', 'figure'),
#     Output('block-heatmap', 'figure'),
#     Input('att-heatmap', 'hoverData'),
#     Input('block-heatmap', 'hoverData'),
#     State('att-heatmap', 'figure'),
#     State('block-heatmap', 'figure'))
# def synchronize_hover(att_hoverData, block_hoverData, att_fig_dict, block_fig_dict):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         return dash.no_update
#
#     trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     try:
#         if trigger_id == 'att-heatmap' and att_hoverData:
#             hovered_token = att_hoverData['points'][0]['hovertext'].split('<br>')[0].split(': ')[1]
#             hovered_layer = int(att_hoverData['points'][0]['y'])
#             hovered_index = int(att_hoverData['points'][0]['x'])
#             print("hovered token on att-heatmap =", hovered_token)
#             print(hovered_layer)
#             print(hovered_index)
#
#             block_hovertext = block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
#             block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index] = f"{block_hovertext}<br>(Hovered in Block Output Map)"
#             # block_hovertext = block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
#             # if block_hovertext not in att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]:
#             #     att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index] += f'<br>{block_hovertext}'
#
#         elif trigger_id == 'block-heatmap' and block_hoverData:
#             hovered_token = block_hoverData['points'][0]['hovertext'].split('<br>')[0].split(': ')[1]
#             hovered_layer = int(block_hoverData['points'][0]['y'])
#             hovered_index = int(block_hoverData['points'][0]['x'])
#             print("hovered token on block-heatmap =", hovered_token)
#             print(hovered_layer)
#             print(hovered_index)
#             # Update att_fig_dict hovertext
#             att_hovertext = att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
#             att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index] = f"{att_hovertext}<br>(Hovered in Attention Map)"
#             # att_hovertext = att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
#             # if att_hovertext not in block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]:
#             #     block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index] += f'<br>{att_hovertext}'
#
#             # Update att_fig_dict with corresponding hover information
#             # att_fig_dict = update_hover_info_for_other_graph(att_fig_dict, hovered_layer, hovered_index)
#
#     except Exception as e:
#         print("Error during hover synchronization:", e)
#         return dash.no_update
#
#     return go.Figure(att_fig_dict), go.Figure(block_fig_dict)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

