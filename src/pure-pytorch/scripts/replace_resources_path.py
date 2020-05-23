def is_useless(line):
    return line[0] == '!' or line.startswith('nltk.download') or line.startswith('from google') or line.startswith('drive.mount')

output = []
with open('src/models/question_generator.py', 'r', encoding='utf-8') as f_in:
    for line in f_in:
        if is_useless(line):
            output.append('# ' + line)
        elif line.startswith('BASE_PATH'):
            output.append("BASE_PATH = ''")
        else:
            output.append(line)

with open('src/models/question_generator.py', 'w', encoding='utf-8') as f_out:
    f_out.writelines(output)
