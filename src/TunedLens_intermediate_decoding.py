import os
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" // for mps
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities import write_to_csv
from tuned_lens.nn.lenses import TunedLens
from tqdm import tqdm
from ModelWrapperBase import BlockOutputWrapper

class TunedLensBlockOutputWrapper(BlockOutputWrapper):
    def __init__(self, block, norm):
        super().__init__(block)
        self.norm = norm
        self.attn_mech_output = None
        self.intermediate_res = None
        self.mlp_output = None
        self.block_output = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.block_output = self.norm(output[0])
        attn_output = self.block.self_attn.activations
        self.attn_mech_output = self.norm(attn_output)
        attn_output += args[0]
        self.intermediate_res = self.norm(attn_output)
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output = self.norm(mlp_output)
        return output

class Tuned_Llama2_Helper:
    def __init__(self, auth_token, model_name_or_path):
        self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=auth_token).to(self.device)
        self.tuned_lens = TunedLens.from_model_and_pretrained(self.model, map_location=self.device)
        # print(self.model)
        # print(self.tuned_lens)

        for i, layer in tqdm(enumerate(self.model.model.layers), desc="Wrapping layers", total=len(self.model.model.layers)):
            self.model.model.layers[i] = TunedLensBlockOutputWrapper(layer, self.model.model.norm)

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_ids.to(self.device)).logits
            return logits

    def forward_with_lens(self, prompt, filename, print_attn_mech=True, print_intermediate_res=True, print_mlp=True,
                          print_block=True):
        self.get_logits(prompt)
        self.model.eval()

        dic = {}
        for idx, layer_wrapper in tqdm(enumerate(self.model.model.layers), desc="Processing layers", total=len(self.model.model.layers), leave=False):
            # assert layer_wrapper.attn_mech_output is not None, "Attention mechanism output is None"
            attn_mech_output = layer_wrapper.attn_mech_output
            intermediate_res = layer_wrapper.intermediate_res
            mlp_output = layer_wrapper.mlp_output
            block_output = layer_wrapper.block_output

            layer_key = f'layer{idx}'
            if layer_key not in dic:
                dic[layer_key] = {
                    'Attention mechanism': [],
                    'Intermediate residual stream': [],
                    'MLP output': [],
                    'Block output': []
                }
            if print_attn_mech:
                dic[layer_key]['Attention mechanism'].extend(self.decode_and_print_top_tokens(attn_mech_output, idx))
            if print_intermediate_res:
                dic[layer_key]['Intermediate residual stream'].extend(self.decode_and_print_top_tokens(intermediate_res, idx))
            if print_mlp:
                dic[layer_key]['MLP output'].extend(self.decode_and_print_top_tokens(mlp_output, idx))
            if print_block:
                dic[layer_key]['Block output'].extend(self.decode_and_print_top_tokens(block_output, idx, True))

        write_to_csv(dic, filename)

    def decode_and_print_top_tokens(self, hidden_states, layer_idx, block_output=False):
        # Transform hidden states into logits
        if block_output == False or layer_idx != 31:
            logits = self.tuned_lens.forward(hidden_states, layer_idx)
            probs = torch.nn.functional.softmax(logits[0][-1], dim=-1)
        else:
            probs = torch.nn.functional.softmax(self.model.lm_head(hidden_states[0][-1]), dim=-1)

        top_probs, top_indices = torch.topk(probs, 10, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(top_indices.squeeze().tolist())
        probabilities = top_probs.squeeze().tolist()
        probs_percent = [round(v * 100, 2) for v in probabilities]
        token_prob_pairs = list(zip(tokens, probs_percent))
        return token_prob_pairs

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()
