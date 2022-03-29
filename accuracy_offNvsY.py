from sklearn.metrics import f1_score
import re
import numpy as np
import pandas as pd

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model-truth", help="comma separated file with an input and output row")
    parser.add_argument("--input-predictions", args="+",  help="comma separated file with an input and output row")
    parser.add_argument("--output", help="the text file to save the accuracy data into")

    args = parser.parse_args()

    y_true = pd.read_csv(args.input_model_truth)
    test_offY_val = list(y_true["offensiveYN"])

    for i in range(len(args.input_predictions)):
        y_pred = pd.read_csv(args.input_predictions[i], index_col=0)

        y_pred.columns = ["input", "generated"]

        pred_offY_val = [0 if string.contains("OffY") else 1 for string in list(y_pred["generated"])]


        assert pred_offY_val == test_offY_val

        sum_int = sum(pred_offY_val == test_offY_val)

        accuracy_score = sum_int / len(test_offY_val)

        f1_val = f1_score(pred_offY_val, test_offY_val)

        with open(args.output[i], 'w') as f:
            f.write("sum of the Offensive YN: \n" + str(sum_int))
            f.write("raw accuracy of the Offensive YN: \n" + str(accuracy_score))
            f.write("f1 accuracy score of Offensive YN: \n" + str(f1_val))

        f.close()