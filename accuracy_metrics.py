from sklearn.metrics import f1_score
import re
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse

def BLEUScores(merged_df_preds):
    # get data within the implication bounds
    list_preds = merged_df_preds["generated"].tolist()
    real_val_list = merged_df_preds["output"].tolist()
    import re
    total_val = []
    for i in range(0, len(list_preds)):
        list_preds[i] = list_preds[i].replace("<pad>", "").replace("[eoo]", "").replace("[cls]", "").replace(
            "[boo]", "")
        real_val_list[i] = real_val_list[i].replace("<pad>", "").replace("[eoo]", "").replace("[cls]", "").replace(
            "[boo]", "")
        if "[ste]" in list_preds[i] and "[ste]" in real_val_list[i]:
            reference = [list_preds[i].split("[ste]")[1].strip().split(" ")]
            candidate = real_val_list[i].split("[ste]")[1].strip().split(" ")
            twogram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0),
                                    smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
            threegram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0),
                                      smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
            total_val.append((twogram + threegram) / 2)
    # 2 and 3 gram mean for all the data
    # compare data to actual implications using BLEU score
    # smoothing method1 - NLTK sentence_bleu add
    # steps: find where the data that takes implications is in the actual data, compare it with the other one
    return np.mean(total_val)


def structural_acc(merged_df_preds):
    total_values = []
    # TODO: add in a check to make sure the prediction is correct before adding it to total_values
    for i in range(0, len(merged_df_preds)):
        if (i % 1000 == 0):
            print(i)
        # find all tokens between square brackets
        tokens_input = re.findall("(?<=\[)[^]]+(?=\])", merged_df_preds["output"].tolist()[i])
        tokens_generated = re.findall("(?<=\[)[^]]+(?=\])", merged_df_preds["generated"].tolist()[i])
        # compare the arrays elementwise to determine number of tokens correctly preserved, divide by total amount of tokens preserved
        try:
            total_values.append((np.array(tokens_input) == np.array(tokens_generated)).sum() / len(tokens_input))
        # short term solution to deal with arrays that are not the same length: talk in meeting about best way to mitigate
        except:
            total_values.append(0);
    return (np.mean(total_values))


def offYAcc(df):

    return sum((("OffY" in row['output'] and "OffY" in row['generated']) or
                ("OffN" in row['output'] and "OffN" in row["generated"]))
               for index, row in df.iterrows()) / len(df)

# Qualitative Testing
def group_testing(df):
    match = []
    for index, row in df.iterrows():
        output_str = row["output"]
        generated_str = row["generated"]
        output = re.search(r'[grp](.*?)[ste]', output_str).group(1)
        generated = re.search(r'[grp](.*?)[ste]', generated_str).group(1)
        if (("OffY" in row['output'] and "OffY" in row['generated']) or
                ("OffN" in row['output'] and "OffN" in row["generated"])):
            match.append(any((output in generate or generate in output) for generate in generated.split()))

    return sum(match) / len(match)

def main(input_predictions):
    df = pd.read_csv(input_predictions, index_col=0).reset_index(drop=True)
    df.columns = ["output", "generated"]
    struc_acc = structural_acc(df)
    bleu_score = BLEUScores(df)
    offy_acc = offYAcc(df)
    group_test = group_testing(df)

    dictionary = dict(zip(["stuctural_acc", "raw_acc_OffYN", "raw_acc_group", "BLEU_acc"], [struc_acc, offy_acc, group_test, bleu_score]))
    print(dictionary)
    #output_df.to_csv(output)
