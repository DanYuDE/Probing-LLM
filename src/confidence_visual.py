import pandas as pd
import ast
import numpy as np
import dash
from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import os

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
class DashApp:
    def __init__(self, sampleText):
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.result_folder = '../result/'  # Update this path if necessary
        self.csv_files = self.get_csv_files(self.result_folder)
        self.file_path = os.path.join(self.result_folder, self.csv_files[0]) if self.csv_files else None
        self.data = pd.read_csv(self.file_path) if self.file_path else pd.DataFrame()
        self.num_layers_per_set = 32
        self.num_sets = len(self.data) // (self.num_layers_per_set + 1) if not self.data.empty else 0
        self.sentences = self.read_sentences('../files/outtest.txt')
        self.sampleText = sampleText
        self.setup_layout()
        self.setup_callbacks()

    def get_csv_files(self, folder_path):
        return [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    def read_sentences(self, file_path):
        with open(file_path, 'r') as file:
            sentences = file.read().strip().split('\n')
        return sentences

    def setup_layout(self):
        self.app.layout = html.Div([
            dcc.Dropdown(
                id='file-selector',
                options=[{'label': file, 'value': file} for file in self.csv_files],
                value=self.csv_files[0] if self.csv_files else None
            ),
            dcc.Dropdown(
                id='set-selector',
                # The options will be generated dynamically in the callback
                value=0
            ),
            html.Div(id='graphs-container'),
        ])

    def setup_callbacks(self):
        @self.app.callback(
            [Output('set-selector', 'options'),
             Output('set-selector', 'value')],
            [Input('file-selector', 'value')])
        def update_set_selector(selected_file):
            if selected_file is not None:
                file_path = os.path.join(self.result_folder, selected_file)
                self.data = pd.read_csv(file_path)  # Update the data
                self.num_sets = len(self.data) // (self.num_layers_per_set + 1)
                # Reset set-selector options and value
                return ([{'label': f'Set {i + 1}', 'value': i} for i in range(self.num_sets)], 0)
            # In case of None or error
            return ([], None)

        @self.app.callback(
            Output('graphs-container', 'children'),
            [Input('set-selector', 'value'),
             Input('file-selector', 'value')])
        def update_graphs(set_num, selected_file):
            if selected_file is None or set_num is None:
                return html.Div()  # Return an empty div if no file is selected or no set number is provided

            # Update file_path and data if a new file is selected
            file_path = os.path.join(self.result_folder, selected_file)
            self.data = pd.read_csv(file_path)

            start_line = set_num * (self.num_layers_per_set + 1)
            attention_weights = self.extractWeightsCSV(self.data, start_line, self.num_layers_per_set,
                                                       'Attention mechanism')
            blockOutput = self.extractWeightsCSV(self.data, start_line, self.num_layers_per_set, 'Block output')

            att_fig = self.create_interactive_heatmap(attention_weights,
                                                      f"Attention Probing Visualization - Set {set_num + 1}")
            block_fig = self.create_interactive_heatmap(blockOutput, f"Block Output Visualization - Set {set_num + 1}")

            sentence = self.sampleText.replace("{{inputText}}", self.sentences[set_num]) if set_num < len(
                self.sentences) else "No sentence available."

            return html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
                html.Div(style={'display': 'flex'}, children=[
                    html.Div(style={'width': '50%', 'marginBottom': '20px'}, children=[
                        dcc.Graph(figure=att_fig, id='att-heatmap'),
                        html.Div(id='att-hover-info', style={'textAlign': 'center', 'marginTop': '10px'})
                    ]),
                    html.Div(style={'width': '50%', 'marginBottom': '20px'}, children=[
                        dcc.Graph(figure=block_fig, id='block-heatmap'),
                        html.Div(id='block-hover-info', style={'textAlign': 'center', 'marginTop': '10px'})
                    ])
                ]),
                html.Div(sentence, style={'width': '100%', 'textAlign': 'left', 'marginTop': '20px'})
            ])

        @self.app.callback(
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
                return dash.no_update

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            att_info = att_info_text if att_info_text is not None else "Hover on Attention Heatmap for info."
            block_info = block_info_text if block_info_text is not None else "Hover on Block Output Heatmap for info."

            if trigger_id == 'att-heatmap' and att_hoverData:
                hovered_layer = int(att_hoverData['points'][0]['y'])
                hovered_index = int(att_hoverData['points'][0]['x'])
                att_hovertext = att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
                att_info = f"Hovered on Attention Heatmap: {att_hovertext.replace('<br>', ', ')}"
                block_hovertext = block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
                block_info = f"Hovered on Block Output Heatmap: {block_hovertext.replace('<br>', ', ')}"

            elif trigger_id == 'block-heatmap' and block_hoverData:
                hovered_layer = int(block_hoverData['points'][0]['y'])
                hovered_index = int(block_hoverData['points'][0]['x'])
                block_hovertext = block_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
                block_info = f"Hovered on Block Output Heatmap: {block_hovertext.replace('<br>', ', ')}"
                att_hovertext = att_fig_dict['data'][0]['hovertext'][hovered_layer][hovered_index]
                att_info = f"Hovered on Attention Heatmap: {att_hovertext.replace('<br>', ', ')}"

            return att_info, block_info

    def extractWeightsCSV(self, data, startLine, numLayers, column_name):
        weight = {}
        for i in range(startLine, startLine + numLayers):
            layer = data.iloc[i]['Layer']
            t_wList = ast.literal_eval(data.iloc[i][column_name])
            tokens, weights = zip(*t_wList)
            weight[layer] = (tokens, weights)
        return weight

    def create_interactive_heatmap(self, tokenWeights, title):
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

    def run(self, debug=True):
        self.app.run_server(debug=debug)

# # Main execution
if __name__ == '__main__':
    my_dash_app = DashApp(sampleText)
    my_dash_app.run()
