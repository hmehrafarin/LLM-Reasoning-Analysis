import pandas as pd
import json

json_file = open("QASC_Dataset/train.json", "r")

data = {"question": [], "answers": [], "fact 1": [], "fact 2": [], "deducted fact": [], "answer": []}

def decapitalize_first_letter(s, upper_rest = False):
  return ''.join([s[:1].lower(), (s[1:].upper() if upper_rest else s[1:])]) 

for line in json_file:
    map_letter2answer = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
    jsonified_line = json.loads(line)
    question = jsonified_line['question']

    answer_template = "(A) {} (B) {} (C) {} (D) {} (E) {} (F) {} (G) {} (H) {}".format(question['choices'][0]['text'],
                                                                                       question['choices'][1]['text'],
                                                                                       question['choices'][2]['text'],
                                                                                       question['choices'][3]['text'],
                                                                                       question['choices'][4]['text'],
                                                                                       question['choices'][5]['text'],
                                                                                       question['choices'][6]['text'],
                                                                                       question['choices'][7]['text'])
    
    formatted_fact1 = jsonified_line['fact1'].capitalize() if jsonified_line['fact2'][-1] != "." else jsonified_line['fact1'].capitalize()[:-1]

    formatted_fact2 = decapitalize_first_letter(jsonified_line['fact2']) if jsonified_line['fact2'][-1] == "." else \
                        decapitalize_first_letter(jsonified_line['fact2']) + "."
    
    composed_fact_template = "{}, and {}".format(formatted_fact1, formatted_fact2)

    formatted_combined_fact = decapitalize_first_letter(jsonified_line['combinedfact']) + "." if jsonified_line['combinedfact'][-1] != "." else \
                                decapitalize_first_letter(jsonified_line['combinedfact'])
    
    deducted_fact_template = "Therefore, {}".format(formatted_combined_fact)

    final_answer_template = "({}) {}".format(jsonified_line['answerKey'], question['choices'][map_letter2answer[jsonified_line['answerKey'].upper()]]['text'])

    data['question'].append(question['stem'])
    data['answers'].append(answer_template)
    data['fact 1'].append(jsonified_line['fact1'])
    data['fact 2'].append(jsonified_line['fact2'])
    # data['composed fact'].append(composed_fact_template)
    data['deducted fact'].append(formatted_combined_fact)
    data['answer'].append(final_answer_template)
    
df = pd.DataFrame(data)

df.to_json("new_train.json", orient ='records')

