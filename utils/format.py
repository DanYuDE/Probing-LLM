def reformat_new_line(line):
    # Splitting the line into the question part and the options part
    if ": Input" in line:
        question_part, options_part = line.split(': Input:', 1)
    elif "Input" in line:
        question_part, options_part = line.split('Input:', 1)

    # Removing the word 'Question:' and trimming spaces
    # if "Question" in line:
    #     question_text = question_part.replace('Question:', '').strip()

    if "Question" in line:
        question_text = question_part.replace('?', '').strip()

    # Trimming the options part and removing any trailing text related to the answer prompt
    options_text = options_part.split(' The correct answer is', 1)[0].strip()

    # Reformatting the line
    formatted_line = f'{question_text}? Options: {options_text} The correct answer: \n'

    return formatted_line

original_file_path = '../files/modified_industry_automation_nAns.txt'  # Replace with your file path
final_file_path = '../files/out.txt'  # Replace with your desired output file path

# Processing each line in the new file and writing to another new file
with open(original_file_path, 'r') as original_file, open(final_file_path, 'w') as final_file:
    for line in original_file:
        formatted_line = reformat_new_line(line)
        final_file.write(formatted_line)