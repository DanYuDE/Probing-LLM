from intermediate_transformer import Llama7BHelper
from intermediate_transformer import clear_csv

token = "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"  # huggingface token

textfile = '../files/out.txt'
outputFile = '../output/new_output3.csv'

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
#          1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13
#          ., ., A, A, ., A, A, D, ., ., D, D, .
# answer : D, B, D, B, B, D, D, B, C, B, A, C, C

#new_output3
#          1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13
#          ., ., ., A, ., A, B, D, ., ., D, D, .
# answer : D, B, D, B, B, D, D, B, C, B, A, C, C
sampleText = """Role and goal:
You are a quizzer to answer the question with given options A, B, C, D, and possibly more. Your primary goal is to output the correct answer clearly by specifying the option letter (A, B, C, or D) that you believe is correct.  
Consider the provided question and its options carefully. Below are two examples of questions with their options and the correct answers for your reference.
Question: Which of the following is a clustering algorithm in machine learning? Options: A.Expectation Maximization B.CART C.Gaussian Naive Bayes D.Apriori. The correct answer:A.
Question: Which is a way the agricultural biotechnology industry could have a positive impact on the environment? Options:A. by producing crops that are virus resistant B.by making robots to replace large farm machines C.by reducing the need for countries to import food D.by increasing the use of wind farms that produce electricity. The correct answer:C.
Now you can start to answer the question with given options to give the correct answer."""
def Execution():
    model = Llama7BHelper(token)
    clear_csv(outputFile)
    textList = []
    with open(textfile, 'r') as file:
        # texts = file.readlines()
        for text in file:
            textList.append(text.rstrip('\n'))
   
        print(textList)

    query = 1
    for t in textList:
        print(query)
        model.decode_all_layers(text=sampleText + "\n" + t,
                               print_intermediate_res=True,
                               print_mlp=True,
                               print_block=True,
                               filename=outputFile)
   
        query += 1


    # lst = ['Question: Which city is the capital of Germany? Options: A.Munich B.Berlin C.Tokyo D.Taipei. The correct answer:',
    #        'Question: In which city is the headquarters of BMW company? Options: A.Taoyuan B.Amsterdam C.Munich D.Shanghai. The correct answer:',
    #        'Question: In whcih city is the MIT? Options: A.New York B.Los Angelos C.Miami D.Cambridge. The correct answer:',
    #        'Question: Which city is the capital of the US? Options: A.Chicago B.Los Angelos C.Pheonix D.Washington. The correct answer:']

    # query = 1
    # for t in lst:
    #     print(query)
    #     model.decode_all_layers(text=sampleText + "\n" + t,
    #                             print_intermediate_res=True,
    #                             print_mlp=True,
    #                             print_block=True,
    #                             filename=outputFile)
    #     query += 1

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



"""
Found:
if the instruction is only one example, the result is wrong -> output.csv
if the instructions are more than one example, the result is correct -> Assumed (output2.csv)
"""
