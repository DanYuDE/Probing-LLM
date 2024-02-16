# Normal Logit-Lens code
import os
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" // for mps
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities import write_to_csv
from ModelWrapperBase import BlockOutputWrapper
from tqdm import tqdm

class LogitBlockOutputWrapper(BlockOutputWrapper):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__(block, unembed_matrix, norm)
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


class Llama7BHelper:
    def __init__(self, token, model_name):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token).to(self.device)
        for i, layer in tqdm(enumerate(self.model.model.layers), desc="Wrapping layers", total=len(self.model.model.layers)):
            self.model.model.layers[i] = LogitBlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)

        self.first_write = True

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
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        # print("values", values)
        # print(values.shape)
        # print("indices", indices)
        # print(indices.shape)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))

        # Convert to list for CSV writing
        decoded_output = [label] + list(zip(tokens, probs_percent))
        return decoded_output

    def decode_all_layers(self, prompt, filename, print_attn_mech=True, print_intermediate_res=True, print_mlp=True,
                          print_block=True):
        self.get_logits(prompt)
        dic = {}
        for i, layer in tqdm(enumerate(self.model.model.layers), desc="Processing layers", total=len(self.model.model.layers), leave=False):
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
                # print(decoded_output[1:])
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

        write_to_csv(dic, filename)