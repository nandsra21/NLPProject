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

    y_test = pd.read_csv(args.input_model_truth)
    y_pred = pd.read_csv(args.input_predictions, index_col=0)["Generated Text"].values

    list_of_data = []
    # split on parenthesis
    for i in range(0, len(y_pred)):
        list_of_data.append(list(filter(lambda x: x != "", [sentence.strip() for sentence in
                                                    y_pred[i].replace("<pad>", '').replace("[", '').replace("boo",'')
                                                    .replace("grp", '').replace("ste", '').replace("eoo", '')
                                                    .split(']')])))

    list_of_data = pd.DataFrame(list_of_data)
    list_of_data[0] = list_of_data[0].apply(lambda label: 1 if label == "OffN" else 0)
    
    sum_int = sum(list_of_data[0].values == y_test["offensiveYN"].values)

    accuracy_score = sum_int / len(y_test["offensiveYN"])

    f1_val = f1_score(y_test["offensiveYN"].values, list_of_data[0].values)

    # Qualitative Testing

    match = []
    for i in range(0, len(list_of_data)):
        if list(y_test["offensiveYN"].values)[i] == 0:
            list_of_data_val = str(list(list_of_data[1].values)[i])
            y_test_val = list(y_test["group"].values)
            match.append(any(list_of_data_val in s for s in y_test_val))
            
    raw_accuracy_group = sum(match) / len(match)


    with open(args.output, 'w') as f:
        f.write("raw accuracy of the Offensive YN: \n" + str(accuracy_score))
        f.write("f1 accuracy score of Offensive YN: \n" + str(f1_val))
        f.write("raw accuracy score of the group tokens: \n" + str(raw_accuracy_group))

    f.close()