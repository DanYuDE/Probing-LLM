import os
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" // for mps
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities import write_to_csv
from tuned_lens.nn.lenses import TunedLens
from tqdm import tqdm

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
    def __init__(self, auth_token, model_name_or_path):
        self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=auth_token).to(self.device)
        self.tuned_lens = TunedLens.from_model_and_pretrained(self.model, map_location=self.device)
        # print(self.model)
        # print(self.tuned_lens)

        for i, layer in tqdm(enumerate(self.model.model.layers), desc="Wrapping layers", total=len(self.model.model.layers)):

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

    def forward_with_lens(self, prompt, filename, print_attn_mech=True, print_intermediate_res=True, print_mlp=True,
                          print_block=True):
        # Ensure the model is in evaluation mode and no gradients are computed
        self.get_logits(prompt)
        self.model.eval()
        # with torch.no_grad():
        #     # Pass input through the model
        #     _ = self.model(input_ids=input_ids)

        # data = []
        dic = {}
        for i, layer_wrapper in tqdm(enumerate(self.model.model.layers), desc="Processing layers", total=len(self.model.model.layers), leave=False):
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
            if print_attn_mech:
                dic[layer_key]['Attention mechanism'].extend(self.decode_and_print_top_tokens(attn_mech_output, i))
            if print_intermediate_res:
                dic[layer_key]['Intermediate residual stream'].extend(self.decode_and_print_top_tokens(intermediate_res, i))
            if print_mlp:
                dic[layer_key]['MLP output'].extend(self.decode_and_print_top_tokens(mlp_output, i))
            if print_block:
                dic[layer_key]['Block output'].extend(self.decode_and_print_top_tokens(block_output, i))

        write_to_csv(dic, filename)

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

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()
