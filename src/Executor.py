from intermediate_transformer import Llama7BHelper
from Tuned import Tuned_Llama7BHelper
from utilities import clear_csv
from confidence_visual import DashApp
import multiprocessing
import webbrowser
import os
import signal

token = "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"  # huggingface token

#              1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12
# output3      ., ., A, A, ., A, A, D, ., ., D, D
# new_output3  ., ., ., A, ., A, B, D, ., ., D, D
# chat-hf      ., ., ., A, A, A, C, D, A, ., C, A

# answer       D, B, D, B, B, D, D, B, C, B, A, C

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

class UserInterface:
    def __init__(self, token, sampleText):
        self.token = token
        self.sampleText = sampleText
        self.commands = {
            "1": self.logit_lens,
            "2": self.tuned_lens,
            "3": self.other_probing,
            "4": self.visualization,
            "sd": self.shutdown
        }
        self.descriptions = ["1. Logit-Lens", "2. Tuned-Lens", "3. Other probing methods", "4. Visualization", "quit (Q or q)"]
        self.dash_app_running = False

    def selectfile(self):
        defaultinput = '../files/outtest.txt'
        defaultoutput = '../output/t.csv'
        inputfile = input("Enter the input file name (including its path) or 0 for default (outtest.txt): ")
        if inputfile.lower() == "q":
            print("Return to main menu...")
            return None
        if inputfile == "0":
            inputfile = defaultinput
        else:
            outputFile = input("Enter the output file name (including its path) or 0 for default (t.csv): ")
            if outputFile.lower == "q":
                print("Return to main menu...")
                return None
            if outputFile == "0":
                outputFile = defaultoutput
        return inputfile, outputFile

    def printCommands(self):
        longestDescriptionLength = 0

        for description in self.descriptions:
            longestDescriptionLength = max(longestDescriptionLength, len(description))

        wave_length = longestDescriptionLength + 2

        print(' ' + "-" * wave_length)
        for description in self.descriptions:
            # Adjust the description to have even spacing and a tab (4 spaces)
            formattedDescription = description.ljust(longestDescriptionLength)
            print(f"| {formattedDescription} |")
        print(' ' + "-" * wave_length)

    def logit_lens(self):
        print("Logit-Lens selected")
        print("Please wait...")
        file_selection = self.selectfile()
        if file_selection is None:  # Check if user chose to quit
            return  # Return early to stop execution and go back to the main menu
        inputfile, outputfile = file_selection
        clear_csv(outputfile)
        textList = self.readTextfile(inputfile)
        model = Llama7BHelper(self.token)
        query = 1
        for t in textList:
            print(query)
            model.decode_all_layers(text=self.sampleText.replace("{{inputText}}", t), filename=outputfile)
            query += 1
        model.reset_all()

    def tuned_lens(self):
        print("Tuned-Lens selected")
        print("Please wait...")
        file_selection = self.selectfile()
        if file_selection is None:  # Check if user chose to quit
            return  # Return early to stop execution and go back to the main menu
        inputfile, outputfile = file_selection
        clear_csv(outputfile)
        textList = self.readTextfile(inputfile)
        model = Tuned_Llama7BHelper(self.token)
        query = 1
        for t in textList:
            print(query)
            model.decode_all_layers(text=self.sampleText.replace("{{inputText}}", t), filename=outputfile)
            query += 1
        model.reset_all()

    def other_probing(self):
        print("other_probing selected")
        print("Please wait...")

    def visualization(self):
        print("visualization selected")
        print("Please wait...")
        # app = DashApp(self.sampleText)
        # app.run()
        self.process = multiprocessing.Process(target=self.run_dash_app)
        self.process.start()
        self.dash_app_running = True
        webbrowser.open_new('http://127.0.0.1:8050/')
        print("Enter 'sd' to shutdown the server.")

    def run_dash_app(self):
        app = DashApp(self.sampleText)
        app.run(debug=False)

    def shutdown(self):
        if self.dash_app_running:
            print("Shutting down the server...")
            if self.process is not None:
                os.kill(self.process.pid, signal.SIGTERM)  # Send the signal to terminate the process
                self.process.join()  # Wait for the process to terminate
            self.dash_app_running = False
            print("Server shut down. Returning to main menu...")
        else:
            print("Dash app is not running.")


    def userInput(self):
        while True:
            self.printCommands()
            userCommand = input("Enter command: ")

            if userCommand == 'sd':
                self.shutdown()
            elif userCommand in self.commands:
                self.commands[userCommand]()
            elif userCommand in ('q', 'Q'):
                if self.dash_app_running:
                    print("Please shut down the server before quitting.")
                else:
                    print("Exit...")
                    exit()
                    break
            else:
                print("Unknown command. Please try again.")

    def readTextfile(self, inputfile):
        textList = []
        with open(inputfile, 'r') as file:
            for text in file:
                textList.append(text.rstrip('\n'))
        return textList


if __name__ == "__main__":
    user = UserInterface(token=token, sampleText=sampleText)
    user.userInput()