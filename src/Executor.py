from intermediate_transformer import Llama7BHelper
from TunedLens_intermediate_decoding import Tuned_Llama2_Helper
from utilities import clear_csv, readTextfile
from confidence_visual import DashApp
import multiprocessing
import webbrowser
import os
import signal
import config
import time

#              1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12
# output3      ., ., A, A, ., A, A, D, ., ., D, D
# new_output3  ., ., ., A, ., A, B, D, ., ., D, D
# chat-hf      ., ., ., A, A, A, C, D, A, ., C, A

# answer       D, B, D, B, B, D, D, B, C, B, A, C

class UserInterface:
    def __init__(self, token, llama_model):
        self.token = token
        self.llama_model = llama_model
        self.selectedText = None
        self.texts = {"1": config.sampleText, "2": config.testText}
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

    def select_text(self):
        text_choice = input("Choose the selectedText to use - '1' for sampleText, '2' for testText: ").strip()
        if text_choice not in ['1', '2']:
            print("Invalid choice. Returning to main menu...")
            return None, None
        return self.texts.get(text_choice, config.sampleText), "sampleText" if text_choice == '1' else "testText"  # Defaults to sampleText if choice is invalid

    def selectfile(self, operation, model_choice, textPar):
        defaultinput = '../files/question.txt'
        model_map = {"1": 'llama-2-7b-hf', "2": 'llama-2-7b-chat-hf'}
        operation_map = {"1": "LogitLens", "2": "TunedLens"}
        modelName = model_map.get(model_choice, "unknown_model")
        operationName = operation_map.get(operation, "unknown_operation")
        defaultoutput = f"../output/{modelName}_{operationName}_{textPar}_res.csv"

        inputfile = None
        if textPar == "testText":
            inputfile = input(f"Enter the input file name (including its path),"
                              f" or 0 for default ({defaultinput}) or 'q' to return to main menu: ").lower()
            if inputfile == "q":
                print("Return to main menu...")
                return None, None
            inputfile = defaultinput if inputfile == "0" else inputfile

        outputFile = input(f"Enter the output file name (including its path) or 0 for default ({defaultoutput}): ").lower()
        if outputFile == "q":
            print("Return to main menu...")
            return None, None
        outputFile = defaultoutput if outputFile == "0" else outputFile

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

        self.selectedText, textPar = self.select_text()
        if self.selectedText is None:
            return
        inputfile, outputfile = self.selectfile("1", model_choice, textPar)
        clear_csv(outputfile)

        model = Llama7BHelper(self.token, self.llama_model[int(model_choice) - 1]) if model_choice in ["1", "2"] else None
        if model is None:
            print("Invalid model choice. Returning to main menu.")
            return

        textList = [self.selectedText] if inputfile is None else readTextfile(inputfile)

        for i, text in enumerate(textList):
            prompt = text if inputfile is None else self.selectedText.replace("{{inputText}}", text)
            print(f"Probing selectedText {i + 1}:" if inputfile else "Probing sample text:")
            startTime = time.time()
            model.decode_all_layers(prompt=prompt, filename=outputfile)
            endTime = time.time() - startTime
            print(f"Time taken: {endTime} seconds")

        model.reset_all()

    def tuned_lens(self, model_choice):
        print("Tuned-Lens selected")
        print("Selected Model:", model_choice)
        print("Please wait...")

        self.selectedText, textPar = self.select_text()
        if self.selectedText is None:
            return
        inputfile, outputfile = self.selectfile("2", model_choice, textPar)
        if inputfile is None or outputfile is None:
            return
        clear_csv(outputfile)

        # Determine model based on choice
        model = Tuned_Llama2_Helper(self.token, self.llama_model[int(model_choice) - 1]) if model_choice in ["1", "2"] else None
        if model is None:
            print("Invalid model choice. Returning to main menu.")
            return

        # Prepare the selectedText list based on inputfile presence
        textList = [self.selectedText] if inputfile is None else readTextfile(inputfile)

        for i, text in enumerate(textList):
            prompt = text if inputfile is None else self.selectedText.replace("{{inputText}}", text)
            print(f"Probing selectedText {i + 1}:" if inputfile else "Probing sample text:")
            startTime = time.time()
            model.forward_with_lens(prompt=prompt, filename=outputfile)
            endTime = time.time() - startTime
            print(f"Time taken: {endTime} seconds")

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
        app = DashApp(config.testText)
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
                modelChoice = input("Enter model choice: ").lower()
                if modelChoice == 'q':
                    print("Returning to main menu...")
                    continue

                if modelChoice in ["1", "2"]:
                    self.logit_lens(modelChoice) if userCommand == '1' else self.tuned_lens(modelChoice)
                else:
                    print("Invalid model choice. Please try again.")

            elif userCommand in ('q', 'Q'):
                print("Exit..." if not self.dash_app_running else "Please shut down the server before quitting.")
                if not self.dash_app_running:
                    break

            elif userCommand in self.commands.keys() and userCommand not in ['1', '2']:
                self.commands[userCommand]()  # Call other commands without model choice
            else:
                print("Unknown command. Please try again.")


if __name__ == "__main__":
    user = UserInterface(token=config.token, llama_model=config.model)
    user.userInput()