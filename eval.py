import pandas as pd
from evaluate import load
from statistics import mean

# bertscore = load("bertscore")

df = pd.read_json('generated_response-full (1 tokens ablated question).json')
# df_2 = pd.read_json('Generated Response/generated_response_QA.json')

count = 0
num_nulls = 0
tp = 0
new_df = []
tp_df = []
generated_deductions = []
actual_deductions = []

for i in range(len(df)):
    count+=1
    # generated_deductions.append(df.iloc[i]['generated deduced'])
    # actual_deductions.append(df.iloc[i]['actual deduced'])
    # Accuracy
    if df.iloc[i]['pred answer'] == "O":
        num_nulls += 1
    if df.iloc[i]['true answer'].lower() == df.iloc[i]['pred answer'].lower() or df.iloc[i]['true answer'].lower()[3:] == df.iloc[i]['pred answer'].lower()[3:]:
        tp += 1
        tp_df.append(df.iloc[i])
    else:
        new_df.append(df.iloc[i])

# results = bertscore.compute(predictions=generated_deductions, references=actual_deductions, lang="en")

new_df = pd.DataFrame(new_df)

tp_df = pd.DataFrame(tp_df)
# unmatched_df = pd.DataFrame(unmatched)
# unmatched_df.to_json("unmatched_QA.json", orient ='records')
# tp_df.to_json("True_positives_full.json", orient ='records')
# new_df.to_json("False_positives_full.json", orient ='records')
print("Number of Instances: {}".format(count))
print("Number of Nulls: {}".format(num_nulls))
print("True Positive: {}".format(tp))
print("Accuracy: {}".format(tp/count))
# print(results['f1'].count(1.0))
# print(mean(results['f1']))

