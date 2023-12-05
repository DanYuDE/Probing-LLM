from intermediate_transformer import Llama7BHelper
from intermediate_transformer import clear_csv
import matplotlib.pyplot as plt
import numpy as np
import torch


token = "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"  # huggingface token

# text = """
#         Instruction: The wheels and gears of a machine are greased in order to decrease what?
#         Input:
#         A.potential energy
#         B.efficiency
#         C.output
#         D.friction
#
#         Output:"""

textfile = '../files/modified_industry_automation_nAns.txt'
outputFile = 'test.csv'

"""
"Follow the question and given input choices to give the correct output answer from the input: "+ text -> filtered_wInstructions.csv
text="Review the question, evaluate the given options, and select an option as the correct answer: "+ text -> new_filtered.csv
"""
def Execution():
    model = Llama7BHelper(token)
    with open(textfile, 'r') as file:
        texts = file.readlines()
        query_id = 1
        clear_csv(outputFile)
        for text in texts:
            model.decode_all_layers(text="Review the question, evaluate the given options, and select an option as "
                                         "the correct answer."
                                         "e.g. Question: Which of the following is a clustering algorithm in machine learning? "
                                         "Input: A. Expectation Maximization B. CART C. Gaussian Naive Bayes D. Apriori. "
                                         "The correct answer is A. "
                                         "Question: The primary focus of the ?????systems approach????? to the problems of business and industry is to improve: "
                                         "Input: A. organizational performance B. work habits C. organizational morale D. individual morale. "
                                         "The correct answer is",
                                    print_intermediate_res=True,
                                    print_mlp=True,
                                    print_block=True,
                                    filename=outputFile, query_id=query_id)

            print(query_id)
            query_id += 1


    # model.reset_all()
    # layer = 14
    # model.get_logits('bananas')
    # attn = model.get_attn_activations(layer)
    # last_token_attn = attn[0][-1]
    # model.set_add_attn_output(layer, 0.6 * last_token_attn)
    #
    # print("\n------------ generate text ------------\n")
    #
    # print(model.generate_text("""Role and goal:
    #     You are a manager of an automation industry, you need to answer the below question. The question has (A), (B), (C), (D) choices.
    #     Question 1: The wheels and gears of a machine are greased in order to decrease what?
    #     A.potential energy
    #     B.efficiency
    #     C.output
    #     D.friction
    #     The Correct Answer is D
    #
    #     Question 2. Toyota's Prius and Honda's hybrid Civic are examples of technological products inspired by
    #     A. Style considerations in the Japanese automobile industry.
    #     B. Social pressure to develop more fuel-efficient vehicles with fewer dangerous emissions.
    #     C. The desire of many engineers to simply make interesting products.
    #     D. The realization that Japanese people didn't need large (D) high-speed cars.
    #     The Correct Answer is B
    #
    #     Question 3. Which of the following is NOT a characteristic of perfectly competitive industry?
    #     A. Free entry into the industry.
    #     B. Product differentiation.
    #     C. Perfectly elastic demand curve.
    #     D. Homogeneous products.
    #     The Correct Answer is""", max_length=20))
    model.reset_all()


if __name__ == "__main__":
    Execution()

# Role and goal: You are a manager of an automation industry, you need to answer the below question. The question has (A), (B), (C), (D) choices.
