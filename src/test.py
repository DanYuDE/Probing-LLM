import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens
import pandas as pd
import config
from tqdm import tqdm
# from utilities import write_to_csv

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

        self.attn_mech_output = None
        self.intermediate_res = None
        self.mlp_output = None
        self.block_output = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        # Skip the unembedding and norm steps as they will be handled by TunedLens
        self.attn_mech_output = self.block.self_attn.activations
        attn_output = self.attn_mech_output
        self.intermediate_res = attn_output + args[0]
        mlp_output = self.block.mlp(self.post_attention_layernorm(self.intermediate_res))
        self.mlp_output = mlp_output
        self.block_output = self.mlp_output + self.intermediate_res
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations

class Tuned_Llama2_Helper:
    def __init__(self, model_name_or_path, auth_token):
        self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=auth_token)
        self.tuned_lens = TunedLens.from_model_and_pretrained(self.model, map_location=self.device)
        self.bar = tqdm(total=100)
        # print(self.model)
        # print(self.tuned_lens)

        for i, layer in enumerate(self.model.model.layers):
            # Initialize the unembed_matrix and norm for each BlockOutputWrapper
            unembed_matrix = self.tuned_lens.layer_translators[i]
            norm = self.model.model.norm
            # Wrap the original layer with BlockOutputWrapper
            self.model.model.layers[i] = BlockOutputWrapper(layer, unembed_matrix, norm)
            # print(f"Layer {i} wrapped: {self.model.model.layers[i]}")

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_ids.to(self.device)).logits
            return logits

    def forward_with_lens(self, prompt):
        # Ensure the model is in evaluation mode and no gradients are computed
        self.get_logits(prompt)
        self.model.eval()
        # with torch.no_grad():
        #     # Pass input through the model
        #     _ = self.model(input_ids=input_ids)

        # data = []
        dic = {}
        for i, layer_wrapper in tqdm(enumerate(self.model.model.layers), desc="Processing layers", total=len(self.model.model.layers)):
            assert layer_wrapper.attn_mech_output is not None, "Attention mechanism output is None"
            attn_mech_output = layer_wrapper.attn_mech_output
            intermediate_res = layer_wrapper.intermediate_res
            mlp_output = layer_wrapper.mlp_output
            block_output = layer_wrapper.block_output

            layer_key = f'layer{i}'
            if layer_key not in dic:
                dic[layer_key] = {
                    'Attention mechanism': [],
                    'Intermediate residual stream': [],
                    'MLP output': [],
                    'Block output': []
                }

            dic[layer_key]['Attention mechanism'].append(self.decode_and_print_top_tokens(attn_mech_output, i))
            dic[layer_key]['Intermediate residual stream'].append(self.decode_and_print_top_tokens(intermediate_res, i))
            dic[layer_key]['MLP output'].append(self.decode_and_print_top_tokens(mlp_output, i))
            dic[layer_key]['Block output'].append(self.decode_and_print_top_tokens(block_output, i))

            # Decode and print top tokens
            # data.append({
            #     "Layer": f'layer{i}',
            #     "Attention": self.decode_and_print_top_tokens(attn_mech_output, i),
            #     "Intermediate": self.decode_and_print_top_tokens(intermediate_res, i),
            #     "MLP": self.decode_and_print_top_tokens(mlp_output, i),
            #     "Block": self.decode_and_print_top_tokens(block_output, i),
            # })
        # df = pd.DataFrame(data)
        # print(df)
        # df.to_csv("layer_outputs.csv", index=False)
        write_to_csv(dic, "layer_outputs.csv")

    def decode_and_print_top_tokens(self, hidden_states, layer_idx):
        logits = self.tuned_lens(hidden_states, layer_idx)
        # print(logits)
        probs = torch.nn.functional.softmax(logits[0][-1], dim=-1)
        top_probs, top_indices = torch.topk(probs, 10, dim=-1)
        # print("top_indices", top_indices)
        # print(top_indices.shape)
        # print("top_probs", top_probs)
        # print(top_probs.shape)
        tokens = self.tokenizer.convert_ids_to_tokens(top_indices.squeeze().tolist())
        probabilities = top_probs.squeeze().tolist()
        probs_percent = [round(v * 100, 2) for v in probabilities]
        token_prob_pairs = list(zip(tokens, probs_percent))
        return token_prob_pairs


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

token = config.token
# Replace 'your_model_name_or_path' with the path to your Llama2 model
helper = Tuned_Llama2_Helper('meta-llama/Llama-2-7b-chat-hf', token)

# Prepare input text
textList = []
with open("../files/question.txt", 'r') as file:
    for text in file:
        textList.append(text.rstrip('\n'))

query = 1
for text in textList:
    print("query", query)
    # input_text = sampleText.replace("{{inputText}}", text)
    # input_ids = helper.tokenizer.encode(input_text, return_tensors='pt')

    # Get the decoded outputs
    helper.forward_with_lens(sampleText.replace("{{inputText}}", text))
    query += 1

# # Interpret the decoded outputs
# for layer_idx, layer in decoded_layers.items():
#     print(f"Layer {layer_idx}:")
#     print(f"Attention: {layer['attention']}")
#     print(f"MLP: {layer['mlp']}")
#     print(f"Block: {layer['block']}")
