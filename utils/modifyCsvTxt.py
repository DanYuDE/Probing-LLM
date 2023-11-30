import csv

# Read from the original CSV and write to the new CSV
def csv_to_txt(inputFile, outputFile):
    textList = []
    with open(inputFile, mode='r', newline='', encoding='utf-8') as infile, \
         open(outputFile, mode='w', newline='', encoding='utf-8') as outfile:
        firstline = infile.readline()
        if "question,choice_A,choice_B,choice_C,choice_D,correct_answer" in firstline:
            next(infile)
        for line in infile:
            # Remove the leading number and period
            line = line.split(". ", 1)[-1]
            # Split the line by commas and exclude the last element (correct answer)
            parts = line.split(',')[:-1]
            # Join the parts back into a string
            formattedLine = ' '.join(parts).replace('\n', '') + ' -- The correct answer is: \n'
            outfile.write(formattedLine)

        print("Conversion complete")

csv_to_txt('../files/industry_automation_wAns.csv', '../files/industry_automation_nNum.txt') # need to be modified