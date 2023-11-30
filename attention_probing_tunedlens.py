import torch
from tuned_lens.nn.lenses import TunedLens, LogitLens
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.plotting import PredictionTrajectory
import matplotlib.pyplot as plt
import tkinter as Tk
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

token = "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"

# Setup device and model
device = torch.device('cpu')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=token).to(device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)

# Initialize lenses
tuned_lens = TunedLens.from_model_and_pretrained(model, map_location=torch.device('cpu'))
logit_lens = LogitLens.from_model(model)


# Function to create a plot
def make_plot(lens, text, layer_stride, statistic, token_range):
    input_ids = tokenizer.encode(text)
    targets = input_ids[1:] + [tokenizer.eos_token_id]

    if len(input_ids) == 0:
        print("Please enter some text.")
        return

    if token_range[0] == token_range[1]:
        print("Please provide valid token range.")
        return

    pred_traj = PredictionTrajectory.from_lens_and_model(
        lens=lens,
        model=model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        targets=targets,
    ).slice_sequence(slice(*token_range))

    fig = getattr(pred_traj, statistic)().stride(layer_stride).figure(
        title=f"{lens.__class__.__name__} ({model.name_or_path}) {statistic}")
    fig.show()
    return fig


# User input
text = input("Enter text: ")
statistic = input("Enter statistic (entropy, cross_entropy, forward_kl): ")
layer_stride = int(input("Enter layer stride: "))
token_range_start = int(input("Enter token range start: "))
token_range_end = int(input("Enter token range end: "))
token_range = [token_range_start, token_range_end]

# Choose lens
lens_choice = input("Choose lens (tuned_lens, logit_lens): ")
lens = tuned_lens if lens_choice == "tuned_lens" else logit_lens

# Generate the plot
make_plot(lens, text, layer_stride, statistic, token_range)