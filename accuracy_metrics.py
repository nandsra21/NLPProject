from sklearn.metrics import f1_score
import re
import numpy as np
import pandas as pd

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model-truth", help="comma separated file with an input and output row")
    parser.add_argument("--input-predictions", help="comma separated file with an input and output row")
    parser.add_argument("--output", help="the text file to save the accuracy data into")

    args = parser.parse_args()

    y_true = pd.read_csv(args.input_model_truth)
    y_pred = pd.read_csv(args.input_predictions, index_col=0)
    y_pred.columns = ["input", "generated"]

    merged_df = real_val.merge(preds, on="input", how="inner")

    y_pred["Source Text"] = y_pred["Source Text"].apply(lambda x: x.replace("[boi] ", "")).apply(
        lambda x: x.replace(" [eoi]", ""))

    y_pred["Generated Text"] = y_pred["Generated Text"].apply(lambda x: x.replace("<pad> ", "")).apply(
        lambda x: x.replace(" <pad>", ""))
    y_pred.columns = ["post", "output"]

    y_data = y_pred["output"].values
    list_of_data = []
    # split on parenthesis
    for i in range(0, len(y_pred)):
        list_of_data.append(list(filter(lambda x: x != "", [sentence.strip() for sentence in
                                                    y_data[i].replace("<pad>", '').replace("[", '').replace("boo",'')
                                                    .replace("grp", '').replace("ste", '').replace("eoo", '')
                                                    .split(']')])))

    list_of_data = pd.DataFrame(list_of_data)
    list_of_data[0] = list_of_data[0].apply(lambda label: 1 if label == "OffN" else 0)
    
    sum_int = sum(list_of_data[0].values == y_true["offensiveYN"].values)

    accuracy_score = sum_int / len(y_true["offensiveYN"])

    f1_val = f1_score(y_true["offensiveYN"].values, list_of_data[0].values)

    # Qualitative Testing

    match = []
    for i in range(0, len(list_of_data)):
        if list(y_true["offensiveYN"].values)[i] == 0:
            list_of_data_val = str(list(list_of_data[1].values)[i])
            y_true_val = list(y_true["group"].values)
            match.append(any(list_of_data_val in s for s in y_true_val))
            
    raw_accuracy_group = sum(match) / len(match)

    list_of_data_preds = []
    list_of_data_real = []
    merged_df["output"] = merged_df["output"].apply(lambda x: x.replace("[boi] ", "")).apply(
        lambda x: x.replace(" [eoi]", ""))

    merged_df["generated"] = merged_df["generated"].apply(lambda x: x.replace("<pad> ", "")).apply(
        lambda x: x.replace(" <pad>", ""))

    y_data_real = merged_df["output"].values
    y_data_preds = merged_df["generated"].values

    # filter out incorrect results
    for i in range(0, len(merged_df)):
        list_of_data_preds.append(list(filter(lambda x: x != "", [sentence.strip() for sentence in
                                                                  y_data_preds[i].replace("<pad>", '').replace("[",
                                                                                                               '').replace(
                                                                      "boo", '')
                                              .replace("grp", '').replace("ste", '').replace("eoo", '')
                                              .split(']')])))
        list_of_data_real.append(list(filter(lambda x: x != "", [sentence.strip() for sentence in
                                                                 y_data_real[i].replace("<pad>", '').replace("[",
                                                                                                             '').replace(
                                                                     "boo", '')
                                             .replace("grp", '').replace("ste", '').replace("eoo", '')
                                             .split(']')])))

    list_of_data_real = pd.DataFrame(list_of_data_real)
    list_of_data_preds = pd.DataFrame(list_of_data_preds)
    list_of_data_real[0] = list_of_data_real[0].apply(lambda label: 1 if label == "OffN" else 0)
    list_of_data_preds[0] = list_of_data_preds[0].apply(lambda label: 1 if label == "OffN" else 0)
    mask = (list_of_data_real[0] == list_of_data_preds[0])


    with open(args.output, 'w') as f:
        f.write("raw accuracy of the Offensive YN: \n" + str(accuracy_score))
        f.write("f1 accuracy score of Offensive YN: \n" + str(f1_val))
        f.write("raw accuracy score of the group tokens: \n" + str(raw_accuracy_group))

    f.close()