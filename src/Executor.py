from intermediate_transformer import Llama7BHelper
from intermediate_transformer import clear_csv

token = "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"  # huggingface token

textfile = '../files/outtest.txt'
outputFile = '../output/test.json'

"""
method 1:
text="Follow the question and given input choices to give the correct output answer from the input: "+ text -> filtered_wInstructions.csv

method 2:
text="Review the question, evaluate the given options, and select an option as the correct answer: "+ text -> new_filtered.csv

method 3:
text="Role and goal:
You are a quizzer to answer the question with given options A, B, C, D or more options. Your goal is to output the correct answer. 
You should take into account the provided question, options. Following is the example of the question and options.
Question:
In electrical resistance welding material of electrode should have?
Options:
A. higher electrical conductivities. B. higher thermal conductivities. C. sufficient strength to sustain high pressure at elevated temperatures. D. all of above
The correct answer: D.
Now you can start to answer the question with given options.
"""
# output3
#          1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12
#          ., ., A, A, ., A, A, D, ., ., D, D
# answer : D, B, D, B, B, D, D, B, C, B, A, C

#new_output3
#          1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12
#          ., ., ., A, ., A, B, D, ., ., D, D
# answer : D, B, D, B, B, D, D, B, C, B, A, C
# sampleText = """Role and goal:
# You are a quizzer to answer the question with given options A, B, C, D, and possibly more. Your primary goal is to output the correct answer clearly by specifying the option letter (A, B, C, or D) that you believe is correct.
# Consider the provided question and its options carefully. Below are two examples of questions with their options and the correct answers for your reference.
# Question: Which of the following is a clustering algorithm in machine learning? Options: A.Expectation Maximization B.CART C.Gaussian Naive Bayes D.Apriori. The correct answer:A.
# Question: Which is a way the agricultural biotechnology industry could have a positive impact on the environment? Options:A. by producing crops that are virus resistant B.by making robots to replace large farm machines C.by reducing the need for countries to import food D.by increasing the use of wind farms that produce electricity. The correct answer:C.
# Now you can start to answer the question with given options to give the correct answer."""

sampleText = """Role and goal: 
Your role is to act as an intelligent problem-solver, tasked with selecting the correct answer from a set of multiple-choice options. Your goal is to carefully analyze the question and each of the provided options, applying your extensive knowledge base and reasoning skills to determine the most accurate and appropriate answer.

Context:
The input text is a question with multiple-choice options. The correct answer is indicated by the option label A, B, C, or D.
1. Question: A clear query requiring a specific answer.
2. Options: A list of possible answers labeled A, B, C, or D.

Instructions:
Analyze the question and options provided.
Use your knowledge to assess each option.
Employ reasoning to eliminate clearly incorrect options.
Identify the most accurate answer based on the information given.
Conclude by justifying your selection, clearly indicating your choice by referencing the option label A, B, C, or D.

Example:
Input: // you will receive the question and options here.
Output: The correct answer:{option label} // you will output the correct answer, replace {option label} with the correct option label A, B, C, or D.

Now you can start to answer the question with given options to give the correct answer.
Input: {{inputText}}
Output: The correct answer:"""

def Execution():
    model = Llama7BHelper(token)
    clear_csv(outputFile)
    textList = []
    with open(textfile, 'r') as file:
        for text in file:
            textList.append(text.rstrip('\n'))
        # print(textList)

    query = 1
    for t in textList:
        print(query)
        model.decode_all_layers(text=sampleText.replace("{{inputText}}", t),
                               print_intermediate_res=True,
                               print_mlp=True,
                               print_block=True,
                               filename=outputFile)
        query += 1
        model.reset_all()

        # print("\n------------ generate text ------------\n")

        # s = model.generate_text(prompt=sampleText.replace("{{inputText}}", t), max_length=350)
        # print("1")
        # print(s)
        # print("2")
        # model.reset_all()
        # query += 1
        # if query == 2:
        #     break

if __name__ == "__main__":
    Execution()


"""
Found:
if the instruction is only one example, the result is wrong -> output.csv
if the instructions are more than one example, the result is correct -> Assumed (output2.csv)
"""
