import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens
import pandas as pd
# from utilities import write_to_csv


class Tuned_Llama2_Helper:
    def __init__(self, model_name_or_path, auth_token):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=auth_token)
        self.tuned_lens = TunedLens.from_model_and_pretrained(self.model, map_location=torch.device('cpu'))
        print(self.model)
        print(self.tuned_lens)
        self.attention_outputs = [None] * len(self.model.model.layers)
        self.intermediate_outputs = [None] * len(self.model.model.layers)
        self.mlp_outputs = [None] * len(self.model.model.layers)
        self.block_outputs = [None] * len(self.model.model.layers)

        self.hooks = []
        for i, layer in enumerate(self.model.model.layers):
            self.hooks.append(layer.self_attn.register_forward_hook(self.capture_attention_output(i)))
            self.hooks.append(layer.mlp.register_forward_hook(self.capture_mlp_output(i)))
            # Block output capturing depends on your definition, here we use the same as MLP for demonstration
            self.hooks.append(layer.mlp.register_forward_hook(self.capture_block_output(i)))

    def capture_attention_output(self, layer_idx):
        def hook(module, input, output):
            self.attention_outputs[layer_idx] = self.tuned_lens.transform_hidden(output[0], layer_idx)

        return hook

    def capture_mlp_output(self, layer_idx):
        def hook(module, input, output):
            self.mlp_outputs[layer_idx] = self.tuned_lens.transform_hidden(output, layer_idx)

        return hook

    def capture_block_output(self, layer_idx):
        def hook(module, input, output):
            self.block_outputs[layer_idx] = output

        return hook

    def clear_captures(self):
        self.attention_outputs = [None] * len(self.model.model.layers)
        self.mlp_outputs = [None] * len(self.model.model.layers)
        self.block_outputs = [None] * len(self.model.model.layers)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward_with_lens(self, input_ids):
        # Ensure the model is in evaluation mode and no gradients are computed
        self.model.eval()
        with torch.no_grad():
            # Pass input through the model. Since hooks are already registered,
            # they will store the required outputs in the respective lists.
            _ = self.model(input_ids=input_ids)
        data = []
        for i, layer in enumerate(self.model.model.layers):
            self.model.eval()
            with torch.no_grad():
                _ = self.model(input_ids=input_ids)


            data.append({
                "Layer": i,
                "Attention": self.decode_and_print_top_tokens(self.attention_outputs[0], f"Attention", i),
                "MLP": self.decode_and_print_top_tokens(self.mlp_outputs[0], f"MLP", i),
                "Block": self.decode_and_print_top_tokens(self.block_outputs[0], f"Block", i),
            })
            df = pd.DataFrame(data)
            df.to_csv("layer_outputs.csv", index=False)
            self.clear_captures()

    def decode_and_print_top_tokens(self, hidden_states, label, layer_idx):
        logits = self.tuned_lens(hidden_states, layer_idx)
        print(logits)
        probs = torch.nn.functional.softmax(logits[0][-1], dim=-1)
        top_probs, top_indices = torch.topk(probs, 10, dim=-1)
        print("top_indices", top_indices)
        print(top_indices.shape)
        print("top_probs", top_probs)
        print(top_probs.shape)
        tokens = self.tokenizer.convert_ids_to_tokens(top_indices.squeeze().tolist())
        probabilities = top_probs.squeeze().tolist()
        probs_percent = [round(v * 100, 2) for v in probabilities]
        token_prob_pairs = list(zip(tokens, probs_percent))
        return token_prob_pairs

    def __del__(self):
        self.remove_hooks()


def write_to_csv(data, filename):
    data_for_df = []
    for layer, layer_data in data.items():
        layer_dict = {
            'Layer': layer,
            'attention': layer_data['attention'],
            # 'Intermediate residual stream': layer_data['Intermediate residual stream'],
            'MLP': layer_data['MLP'],
            'Block': layer_data['Block']
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

token = "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"
# Replace 'your_model_name_or_path' with the path to your Llama2 model
helper = Tuned_Llama2_Helper('meta-llama/Llama-2-7b-chat-hf', token)

# Prepare input text
textList = []
with open("../files/question.txt", 'r') as file:
    for text in file:
        textList.append(text.rstrip('\n'))

input_text = sampleText.replace("{{inputText}}", textList[0])
input_ids = helper.tokenizer.encode(input_text, return_tensors='pt')

# Get the decoded outputs
decoded_layers = helper.forward_with_lens(input_ids)

# Interpret the decoded outputs
for layer_idx, layer in decoded_layers.items():
    print(f"Layer {layer_idx}:")
    print(f"Attention: {layer['attention']}")
    print(f"MLP: {layer['mlp']}")
    print(f"Block: {layer['block']}")
