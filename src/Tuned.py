import os
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" // for mps
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities import write_to_csv
from tuned_lens.nn.lenses import TunedLens

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
        self.activations = output[0]  # (1*403*4096)
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        # Keep a reference to the original MLP to use later
        self.original_mlp = self.block.mlp
        self.block.mlp = AttnWrapper(
            self.block.mlp)  # Optional: Wrap MLP if you want to capture its raw output separately

        # Attributes to store outputs
        self.raw_attn_output = None
        self.raw_intermediate_output = None  # Store the intermediate output
        self.raw_mlp_output = None
        self.raw_block_output = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        # Assuming input_layernorm is applied to the input
        normed_input = self.block.input_layernorm(args[0])

        # Process attention mechanism
        attn_output = self.block.self_attn(normed_input, **kwargs)
        self.raw_attn_output = attn_output[0]

        # Calculate and store intermediate output (after attention, before MLP)
        # Here you may want to apply post_attention_layernorm before the MLP
        intermediate_output = attn_output[0] + normed_input
        self.raw_intermediate_output = self.block.post_attention_layernorm(intermediate_output)

        # Process MLP
        mlp_output = self.original_mlp(self.raw_intermediate_output)
        self.raw_mlp_output = mlp_output[0]

        # For raw block output, assuming it's the output after MLP and residual connection
        block_output = mlp_output[0] + self.raw_intermediate_output
        self.raw_block_output = block_output

        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations


class Tuned_Llama7BHelper:
    def __init__(self, token):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=token).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer)
        self.tuned_lens = TunedLens.from_model_and_pretrained(self.model, map_location=torch.device('cpu'))

        # Inspect the structure of the first decoder layer
        # print(self.model.model.layers[0].block)

        # Optionally, if you need to inspect the entire layer, you might find this useful:
        # print(dir(self.model.model.layers[0].block))  # Lists all attributes and methods

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

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations, dim=-1)
        values, indices = torch.topk(softmaxed, 10)

        # Flatten the values if they're not already flat
        flat_values = values.flatten().tolist()  # Ensure it's a flat list

        probs_percent = [int(v * 100) for v in flat_values]
        tokens = self.tokenizer.batch_decode(indices.flatten().unsqueeze(-1))

        decoded_output = [label] + list(zip(tokens, probs_percent))
        return decoded_output

        # softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        # values, indices = torch.topk(softmaxed, 10, dim=-1)  # Ensure we're taking top 10 across the correct dimension
        # print("value shape:", values.shape)
        # print("indices shape:", indices.shape)
        # tokens = self.tokenizer.batch_decode(indices.squeeze())
        # probs_percent = [round(v * 100, 2) for v in values.squeeze().tolist()]
        #
        #
        # decoded_output = list(zip(tokens, probs_percent))
        # return decoded_output

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

            # print(f"{i} attn: ", layer.attn_mech_output_unembedded.shape)
            # print(f"{i} intermediate: ", layer.intermediate_res_unembedded.shape)
            # print(f"{i} mlp: ", layer.mlp_output_unembedded.shape)
            # print(f"{i} block: ", layer.block_output_unembedded.shape)  # (1*403*32000)

            attn_activations = layer.raw_attn_output
            if attn_activations is not None:
                transformed_attn = self.tuned_lens.transform_hidden(attn_activations, i)
                print(transformed_attn.shape)
                decoded_output = self.print_decoded_activations(transformed_attn, 'Attention mechanism')
                dic[layer_key]['Attention mechanism'].extend(decoded_output[1:])

            intermediate_output = layer.raw_intermediate_output
            if intermediate_output is not None:
                transformed_intermediate = self.tuned_lens.transform_hidden(intermediate_output, i)
                print("ts: ", transformed_intermediate.shape)
                print("ts v: ", transformed_intermediate)
                decoded_output = self.print_decoded_activations(transformed_intermediate, 'Intermediate residual stream')
                dic[layer_key]['Intermediate residual stream'].extend(decoded_output[1:])

                # Process raw MLP output
            mlp_activations = layer.raw_mlp_output
            if mlp_activations is not None:
                transformed_mlp = self.tuned_lens.transform_hidden(mlp_activations, i)
                decoded_output = self.print_decoded_activations(transformed_mlp, 'MLP output')
                dic[layer_key]['MLP output'].extend(decoded_output[1:])

                # Process raw block output
            block_output = layer.raw_block_output
            if block_output is not None:
                transformed_block = self.tuned_lens.transform_hidden(block_output, i)
                decoded_output = self.print_decoded_activations(transformed_block, 'Block output')
                dic[layer_key]['Block output'].extend(decoded_output[1:])

        write_to_csv(dic, filename)
