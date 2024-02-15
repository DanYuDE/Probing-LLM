# Interpretability Study of Large Language Models with Probing Techniques

This research project explores the interpretability of large language models (Llama-2-7B) 
through the implementation of two probing techniques -- Logit-Lens and Tuned-Lens. By dissecting 
the internal workings of these models, I aim to shed light on how they understand and generate 
language, offering insights into their decision-making processes.

## Description

The project delves into the black box of Llama-2-7B model to understand the mechanics 
behind its language understanding capabilities. Using probing techniques, 
I analyze the models at different layers and stages of extracting interpretable features. 

## Getting Started

### Dependencies

- **Hardware:** M1 Macbook Pro
- **OS:** MacOS 14.2.1 (Linux and Windows should work as well, but not tested yet)
- **Python Version:** 3.10
- **Framework:** Pytorch
- **Python Packages:** transformers, pandas, tqdm, numpy, dash, torch

### Installing

Follow these steps to set up a local development environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/DanYuDE/Research-Project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Research-Project
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
   or Using Anaconda:
   ```bash
    conda create -n your_env_name python=3.10
    conda activate your_env_name
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5. Add new `config.py` file, the `config_example.py` file is provided as a template:
    ```bash
    nano config.py
    ```
    Update the `token` (Huggingface token) and `sampleText` (can add more LLM models in `model`).

6. Run the project:
    ```bash
    make
    ```

## Credits and Acknowledgments

### Code Reference

This project incorporates code and techniques inspired by the work of [nrimsky](https://github.com/nrimsky) as detailed in the [Intermediate Decoding Notebook](https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb). We extend our gratitude for the foundational methodologies and code examples provided, which have been pivotal in the development of our probing techniques for LLM interpretability.

### Special Thanks

- I express our sincere appreciation to [nrimsky](https://github.com/nrimsky) for her groundbreaking work on language model interpretability, which has significantly influenced this project.
- Acknowledgment goes to the [transformers](https://github.com/huggingface/transformers) library by Hugging Face, which has been instrumental in facilitating the research.

