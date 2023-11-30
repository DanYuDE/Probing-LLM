import csv

def is_header(line):
    # Define expected headers
    expected_headers = ["question","choice_A","choice_B","choice_C","choice_D","correct_answer"]

    return all(item in [word.strip() for word in line.split(',')] for item in expected_headers)
    # ----- Or -----
    # line_parts = [word.strip() for word in line.split(',')]
    
    # # Check each expected header against the line parts
    # for item in expected_headers:
    #     print(f'{i}: ')
    #     if item not in line_parts:
    #         return False
    #     print("True")

    # return True

# Read from the original CSV and write to the new CSV
def csv_to_txt(inputFile, outputFile, rmNumFlag:bool = False):
    textList = []
    with open(inputFile, mode='r', newline='', encoding='utf-8') as infile, \
         open(outputFile, mode='w', newline='', encoding='utf-8') as outfile:
        firstline = infile.readline()
        
        # Check if the first line is a header
        if not is_header(firstline):
            # If it's not a header, reset the file pointer to the beginning
            print("True")
            infile.seek(0)
            
        for line in infile:
            # Remove the leading number and period
            if rmNumFlag is True:
            	line = line.split(". ", 1)[-1]
            # Split the line by commas and exclude the last element (correct answer)
            parts = line.split(',')[:-1]
            # Join the parts back into a string
            formattedLine = ' '.join(parts).replace('\n', '') + ' ... The correct answer is: \n'
            outfile.write(formattedLine)

        print("Conversion complete")

csv_to_txt('../files/industry_automation.txt', '../files/industry_automation_nAns.txt') # need to be modified
