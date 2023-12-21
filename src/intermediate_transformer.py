# import os
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" // for mps
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        # print("Forward method called in AttnWrapper")
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,) + output[1:]
        self.activations = output[0]
        # print("Activations shape:", self.activations.shape)
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        attn_output += args[0]
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations


class Llama7BHelper:
    def __init__(self, token):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # self.device = torch.device("mps")
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)

        self.first_write = True

    def generate_text(self, prompt, max_length=100): #, temperature=1):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        print("Tokens:", self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length) #, temperature=temperature)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_ids.to(self.device)).logits
            return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        layer_output = self.model.model.layers[layer].get_attn_activations()
        return layer_output
        # if layer_output is not None:
        #     attentions = layer_output.view(1, 60, 64, 64)
        #     return attentions
        # else:
        #     raise ValueError(f"Attention output for layer {layer} is None or invalid.")

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        # print(label, list(zip(tokens, probs_percent)))

        # Convert to list for CSV writing
        decoded_output = [label] + list(zip(tokens, probs_percent))
        return decoded_output

    def decode_all_layers(self, text, filename, print_attn_mech=True, print_intermediate_res=True, print_mlp=True,
                          print_block=True):
        self.get_logits(text)
        dic = {}
        for i, layer in enumerate(self.model.model.layers):
            # print(f'Layer {i}: Decoded intermediate outputs')
            layer_key = f'layer{i}'
            if layer_key not in dic:
                dic[layer_key] = {
                    'Attention mechanism': [],
                    'Intermediate residual stream': [],
                    'MLP output': [],
                    'Block output': []
                }
            if print_attn_mech:
                decoded_output = self.print_decoded_activations(layer.attn_mech_output_unembedded,
                                                                'Attention mechanism')
                dic[f'layer{i}']['Attention mechanism'].extend(decoded_output[1:])
            if print_intermediate_res:
                decoded_output = self.print_decoded_activations(layer.intermediate_res_unembedded,
                                                                'Intermediate residual stream')
                dic[f'layer{i}']['Intermediate residual stream'].extend(decoded_output[1:])
            if print_mlp:
                decoded_output = self.print_decoded_activations(layer.mlp_output_unembedded, 'MLP output')
                dic[f'layer{i}']['MLP output'].extend(decoded_output[1:])
            if print_block:
                decoded_output = self.print_decoded_activations(layer.block_output_unembedded, 'Block output')
                dic[f'layer{i}']['Block output'].extend(decoded_output[1:])

        write_to_json(dic, filename)


def clear_csv(filename):
    # Clear the contents of the CSV file
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
    # empty_row = pd.DataFrame({col: [''] for col in df.columns}, index=[df.index[-1] + 1])
    # df = pd.concat([df, empty_row], ignore_index=True)
    df.set_index('Layer', inplace=True)  # Set the 'Layer' column as the index
    df.to_csv(filename, mode='a', index=True)


def write_to_json(data, filename):
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

    # Convert the list of dictionaries into a JSON string
    json_string = json.dumps(data_for_df, indent=4)
    print(json_string)

    # Write the JSON string to a file
    with open(filename, 'a') as file:
        file.write(json_string)


   # def plot_attention_heatmap(self, attention, tokens, layer_num, head_num, ax):
    #     sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap='viridis', ax=ax)
    #     ax.set_title(f'Layer {layer_num} Head {head_num}')
    #
    # def visualize_attention(self, text, layers=[0], heads=[0]):
    #     inputs = self.tokenizer(text, return_tensors="pt")
    #     tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    #
    #     outputs = self.model(**inputs.to(self.device), output_attentions=True)
    #     attentions = outputs.attentions
    #
    #     fig, axes = plt.subplots(len(layers), len(heads), figsize=(15, 15))
    #     for i, layer in enumerate(layers):
    #         for j, head in enumerate(heads):
    #             ax = axes[i][j] if len(layers) > 1 else axes[j]
    #             attention = attentions[layer][0, head].detach().cpu().numpy()
    #             self.plot_attention_heatmap(attention, tokens, layer, head, ax)
    #     plt.tight_layout()
    #     plt.show()
    #
    # def plot_attention(self, attention_activations, tokens, head_index=0):
    #     # The number of tokens is the square root of the attention matrix size for one head
    #     seq_length = int(np.sqrt(attention_activations.shape[-1]))
    #     attn_matrix = attention_activations[0, head_index].reshape(seq_length, seq_length)
    #
    #     # attn_matrix_norm = (attn_matrix - attn_matrix.min()) / (attn_matrix.max() - attn_matrix.min())
    #
    #     plt.figure(figsize=(10, 8))
    #     ax = sns.heatmap(attn_matrix, xticklabels=tokens[:seq_length], yticklabels=tokens[:seq_length], cmap='viridis',
    #                      linewidths=.5)  # Create a heatmap using seaborn
    #
    #     plt.xticks(rotation=90, fontsize=8)
    #     plt.yticks(rotation=0, fontsize=8)
    #
    #     cbar = ax.collections[0].colorbar
    #     cbar.ax.tick_params(labelsize=10)
    #
    #     plt.title(f"Attention Map for Head {head_index}")
    #     plt.xlabel("Key Tokens")
    #     plt.ylabel("Query Tokens")
    #
    #     plt.tight_layout()
    #     plt.show()
