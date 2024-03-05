import pandas as pd
import evaluate
from statistics import mean
import fire

def eval(
        output_file: str = "",
        data_type: str = "QASC", # Options: QASC, Bamboogle
        model_type: str = "llama", # Options: llama, flan-t5
        write_output_to_file: bool = False,     
        metric: str = "accuracy", # Options: accuracy, rouge, accuracy (no mc)
        jibberish: bool = False,
):
    df = pd.read_json(output_file)
    count = 0
    num_nulls = 0
    tp = 0
    fp_df = []
    tp_df = []
    if metric == "accuracy":
        for i in range(len(df)):
            count+=1
            df.iloc[i]['pred answer'] = df.iloc[i]['pred answer'].rstrip()
            if df.iloc[i]['pred answer'] == "O":
                num_nulls += 1
            if data_type == "Bamboogle":
                if df.iloc[i]['pred answer'].lower() == df.iloc[i]['true answer'].lower():
                    tp += 1
                    tp_df.append(df.iloc[i])
            elif model_type == "llama" and data_type == "QASC":
                if df.iloc[i]['true answer'].lower() == df.iloc[i]['pred answer'].lower() or df.iloc[i]['true answer'].lower()[3:] == df.iloc[i]['pred answer'].lower()[3:]:
                  tp += 1
                  tp_df.append(df.iloc[i])
            elif model_type == "flan-t5" and data_type == "QASC":
                if df.iloc[i]['true answer'].lower() == df.iloc[i]['pred answer'].lower() or df.iloc[i]['true answer'].lower()[:3] == df.iloc[i]['pred answer'].lower()[:3]:
                    tp += 1
                    tp_df.append(df.iloc[i])                 
            else:
                fp_df.append(df.iloc[i]) 

        fp_df = pd.DataFrame(fp_df)
        tp_df = pd.DataFrame(tp_df)
        if write_output_to_file:
            tp_df.to_json("True_positives_full.json", orient ='records')
            fp_df.to_json("False_positives_full.json", orient ='records')

        print("Number of Instances: {}".format(count))
        print("Number of Nulls: {}".format(num_nulls))
        print("True Positive: {}".format(tp))
        print("Accuracy: {}".format(tp/count))

    elif metric == "rouge":
        rouge = evaluate.load('rouge')
        predictions = df['pred answer']
        gold = df['jibberish answer'] if jibberish else df['true answer']
        rouge_scores = rouge.compute(predictions=predictions, references=gold)
        print(rouge_scores)

    elif metric == "accuracy (no mc)":
        for i in range(len(df)):
            count+=1
            if model_type == "llama" and data_type == "QASC":
                if df.iloc[i]['true answer'].lower()[4:] in df.iloc[i]['pred answer'].lower()[4:]:
                    tp += 1
            else:
                if df.iloc[i]['true answer'].lower()[4:] in df.iloc[i]['pred answer'].lower():
                    print(df.iloc[i]['true answer'].lower()[3:])
                    tp += 1

        print("Number of Instances: {}".format(count))      
        print("True Positive: {}".format(tp))
        print("Accuracy: {}".format(tp/count))

if __name__ == "__main__":
    fire.Fire(eval)

