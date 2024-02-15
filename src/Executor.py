from intermediate_transformer import Llama7BHelper
from TunedLens_intermediate_decoding import Tuned_Llama2_Helper
from utilities import clear_csv, readTextfile
from confidence_visual import DashApp
import multiprocessing
import webbrowser
import os
import signal
import config

#              1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12
# output3      ., ., A, A, ., A, A, D, ., ., D, D
# new_output3  ., ., ., A, ., A, B, D, ., ., D, D
# chat-hf      ., ., ., A, A, A, C, D, A, ., C, A

# answer       D, B, D, B, B, D, D, B, C, B, A, C

class UserInterface:
    def __init__(self, token, llama_model, sampleText):
        self.token = token
        self.llama_model = llama_model
        self.sampleText = sampleText
        self.commands = {
            "1": self.logit_lens,
            "2": self.tuned_lens,
            "3": self.other_probing,
            "4": self.visualization,
            "sd": self.shutdown
        }
        self.descriptions = ["1. Logit-Lens", "2. Tuned-Lens", "3. Other probing methods", "4. Visualization", "quit (Q or q)"]
        self.model_choices = ["1. meta-llama/Llama-2-7b-hf", "2. meta-llama/Llama-2-7b-chat-hf", "quit(Q or q)"]
        self.dash_app_running = False

    def selectfile(self, operation, model_choice):
        defaultinput = '../files/question.txt'
        # Construct default output file name based on operation and model choice
        model_map = {"1": 'llama-2-7b-hf', "2": 'llama-2-7b-chat-hf'}
        operation_map = {"1": "LogitLens", "2": "TunedLens"}
        modelName = model_map.get(model_choice, "unknown_model")
        operationName = operation_map.get(operation, "unknown_operation")
        defaultoutput = f"../output/{modelName}_{operationName}_res.csv"

        inputfile = input(f"Enter the input file name (including its path) or 0 for default ({defaultinput}): ")
        if inputfile.lower() == "q":
            print("Return to main menu...")
            return None
        if inputfile == "0":
            inputfile = defaultinput

        outputFile = input(f"Enter the output file name (including its path) or 0 for default ({defaultoutput}): ")
        if outputFile.lower() == "q":
            print("Return to main menu...")
            return None
        if outputFile == "0":
            outputFile = defaultoutput

        return inputfile, outputFile

    def printCommands(self, commands_list):
        longestDescriptionLength = 0

        for description in commands_list:
            longestDescriptionLength = max(longestDescriptionLength, len(description))

        wave_length = longestDescriptionLength + 2

        print(' ' + "-" * wave_length)
        for description in commands_list:
            # Adjust the description to have even spacing and a tab (4 spaces)
            formattedDescription = description.ljust(longestDescriptionLength)
            print(f"| {formattedDescription} |")
        print(' ' + "-" * wave_length)

    def logit_lens(self, model_choice):
        print("Logit-Lens selected")
        print("Selected Model:", model_choice)
        print("Please wait...")
        file_selection = self.selectfile("1", model_choice)
        if file_selection is None:
            return
        inputfile, outputfile = file_selection
        clear_csv(outputfile)
        textList = readTextfile(inputfile)

        # Instantiate the model based on the model_choice
        if model_choice == "1":
            model = Llama7BHelper(self.token, self.llama_model[0])  # Llama-2-7b-hf
        elif model_choice == "2":
            model = Llama7BHelper(self.token, self.llama_model[1])  # Llama-2-7b-chat-hf
        else:
            print("Invalid model choice. Returning to main menu.")
            return

        for i, text in enumerate(textList):
            print(f"Prompt {i + 1}:")
            model.decode_all_layers(prompt=self.sampleText.replace("{{inputText}}", text), filename=outputfile)
        model.reset_all()

    def tuned_lens(self, model_choice):
        print("Tuned-Lens selected")
        print("Selected Model:", model_choice)
        print("Please wait...")
        file_selection = self.selectfile("2", model_choice)
        if file_selection is None:  # Check if user chose to quit
            return  # Return early to stop execution and go back to the main menu
        inputfile, outputfile = file_selection
        clear_csv(outputfile)
        textList = readTextfile(inputfile)
        # Instantiate the model based on the model_choice
        if model_choice == "1":
            model = Tuned_Llama2_Helper(self.token, self.llama_model[0])
        elif model_choice == "2":
            model = Tuned_Llama2_Helper(self.token, self.llama_model[1])
        else:
            print("Invalid model choice. Returning to main menu.")
            return

        for i, text in enumerate(textList):
            print(f"Prompt {i + 1}:")
            model.forward_with_lens(prompt=self.sampleText.replace("{{inputText}}", text), filename=outputfile)
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
            self.printCommands(self.descriptions)  # Initially print main commands
            userCommand = input("Enter command: ")

            if userCommand == 'sd':
                self.shutdown()
            elif userCommand in ['1', '2']:  # For options requiring model choice
                print("Select a model:")
                self.printCommands(self.model_choices)
                modelChoice = input("Enter model choice: ")
                if modelChoice.lower() in ('q', 'Q'):
                    print("Returning to main menu...")
                    continue
                # Convert model choice to the required format
                if modelChoice == "1" or modelChoice == "2":
                    if userCommand == '1':
                        self.logit_lens(modelChoice)
                    elif userCommand == '2':
                        self.tuned_lens(modelChoice)
                else:
                    print("Invalid model choice. Please try again.")
            elif userCommand in ('q', 'Q'):
                if self.dash_app_running:
                    print("Please shut down the server before quitting.")
                else:
                    print("Exit...")
                    exit()
                    break
            elif userCommand in self.commands.keys() and userCommand not in ['1', '2']:
                self.commands[userCommand]()  # Call other commands without model choice
            else:
                print("Unknown command. Please try again.")


if __name__ == "__main__":
    user = UserInterface(token=config.token, llama_model=config.model, sampleText=config.sampleText)
    user.userInput()