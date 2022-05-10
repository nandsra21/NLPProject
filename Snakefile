from snakemake.utils import min_version
min_version("6.0")

#MODEL_VAL = ["20_"+str(5e-3)]#,"20_"+str(3e-3), "50_"+str(3e-3)]
    #, "500_"+str(3e-3), "50_"+str(3e-2), "100_"+str(3e-2), "200_"+str(3e-2), "500_"+str(3e-2)]
EPOCHS = [20]#5, 10, 15, 20] #"10", "20", "50"]
LR = [str(7e-3)]#, str(5e-3), str(6e-3)]
SAMPLES = ["100K"]#10K", "50K", "100K"]
# opening the file in read mode
# my_file = open("model_5/checkpoints.txt","r")
# # reading the file
# data = my_file.read()
# # replacing end splitting the text
# # when newline ('\n') is seen.
# CHECKPOINTS = data.split("\n./")
# CHECKPOINTS[0] = CHECKPOINTS[0][2:]
# CHECKPOINTS[len(CHECKPOINTS) - 1] =  CHECKPOINTS[len(CHECKPOINTS) - 1][:len(CHECKPOINTS[len(CHECKPOINTS) - 1]) - 1]
# print(CHECKPOINTS)
# my_file.close()

rule all:
    input:
        #"predictions_initial_test_10.csv"
        model = expand('model_test_grid_search_8/final/model_{num}_{lr}_{epoch}.pth', num=SAMPLES, lr=LR, epoch=EPOCHS)
        #expand('predictions_{version}_4.csv', version=MODEL_VAL),
        #expand("accuracy_values_{version}_{checkpoint}.txt", version=MODEL_VAL, checkpoint = CHECKPOINTS)
        #expand("accuracy_values_{version}.csv", version=MODEL_VAL)

# --- TODO: make rule to create data after cleaning up jupyter notebook, add NLG demo, add accuracy script --- #

## rule create_data:
##    message: "creating dataframe in the correct form"
##    input:
##    input:
##        train = "SBIC.v2.trn.csv",
##        dev = "SBIC.v2.dev.csv",
##        test = "SBIC.v2.tst.csv"
##    output:
##        train = expand("dataset_{version}", version=MODEL_VAL),
##        dev = expand("dataset_{version}", version=MODEL_VAL),
##        test = expand("dataset_{version}", version=MODEL_VAL)
##    params:
        # --- pick from regular, scrambled, even split --- #
##        type_of_data = "regular"
##    notebook:
##        "csv_manipulate_final.ipynb"
rule create_model:
    input:
        training_csv = expand("sample.{num}.csv", num = SAMPLES),
        validation_csv = expand("sample.{num}.dev.csv", num = SAMPLES),
    output:
        model = expand('model_test_grid_search_8/final/model_{num}_{lr}_{epoch}.pth', num=SAMPLES, lr=LR, epoch=EPOCHS),
        #predictions_initial = expand('model_test_grid_search_8/model_{num}_{lr}_{epoch}/predictions/predictions.csv"', num=SAMPLES, lr=LR, epoch=EPOCHS)
    shell:
        """
        python3 classifier_pytorch.py \
            --input-training {input.training_csv} \
            --input-validation {input.validation_csv} \
            --output {output.model}
        """


# Generation of predictions:

#rule generate_dataframe:
#    input:
#        df = "SBIC.dev.scramble.4.csv",
#    output:
#        directory("predictions_dfs"),
#        file = "predictions_dfs/input_preds_0.csv"
#    run:
#        """
#            import numpy as np
#            import pandas as pd
#            # get batch size value
#            BATCH_SIZE = 32
#            # create separate batched dataframes
#            sampled_df = pd.read_csv("SBIC.dev.scramble.4.csv").sample(frac=1)
#            val = int(float(len(sampled_df)) / 32)
#            list_df = np.array_split(sampled_df, val)
#            for i, df in enumerate(list_df, 1):
#                df.to_csv(f'predictions_dfs/input_preds_{i}.csv')
#        """

#
# import numpy as np
# import pandas as pd
# # get batch size value
# BATCH_SIZE = 32
# # create separate batched dataframes
# sampled_df = pd.read_csv("SBIC.dev.scramble.4.csv").sample(frac=1)
# val = int(float(len(sampled_df)) / 32)
# list_df = np.array_split(sampled_df, val)
# for i, df in enumerate(list_df, 1):
#     df.to_csv(f'predictions_dfs/input_preds_{i}.csv')
# IDS = [f"input_preds_{i}" for i in range(1,val)]
# pass into predictions

#rule generate_checkpoints:
#    input:
#        model = rules.create_model.output.model
#    output:
#        text = checkpoints.txt
#    shell:
#        """
#        cd {input.model} \
#        find -name "checkpoint-*" > {output.text}
#        """


#rule generate_predictions:
# input:
#        model_parent = rules.create_model.output.model,
#        model = expand("model_5/{checkpoint}", checkpoint = CHECKPOINTS),
#        #check = rules.generate_dataframe.output,
#        validation_csv = ancient(expand("predictions_dfs/{id}.csv", id = IDS))
#    output:
#        predictions = expand('predictions_{version}/{id}_{checkpoint}.csv', checkpoint = CHECKPOINTS, version=MODEL_VAL, id = IDS)
#    shell:
#        """
#        python3 predictions.py \
#            --parent-model {input.model_parent} \
#            --input-model {input.model} \
#            --input-validation {input.validation_csv} \
#            --output {output.predictions}
#        """

# rule generate_predictions:
#     input:
#         model_parent = rules.create_model.output.model,
#         model = "model_test_4/model_test_4.pth",
#         validation_csv = "sample.50.dev.csv"
#     output:
#         predictions = expand('predictions_{version}_4.csv', version=MODEL_VAL),
#         predictions_initial = "predictions_initial_test_4.csv"
#     shell:
#         """
#         python3 predictions_pytorch.py \
#             --parent-model {input.model_parent} \
#             --input-model {input.model} \
#             --input-validation {input.validation_csv} \
#             --predictions-initial{output.predictions_initial} \
#             --output {output.predictions}
#         """


# take all generated dataframes and concatenate
# rule aggregate_prediction:
#     input:
#         check = rules.generate_predictions.output,
#         tables=expand('predictions_{version}/{id}_{checkpoint}.csv', checkpoint = CHECKPOINTS, version=MODEL_VAL, id = IDS)
#     output:
#         table= expand('predictions_{version}_{checkpoint}.csv', version=MODEL_VAL, checkpoint = CHECKPOINTS)
#     params:
#         splitOn = val
#     shell:
#         """
#         python3 concatenate_tables.py \
#             --tables {input.tables} \
#             --split {params.splitOn} \
#             --output {output.table}
#         """
#
# rule get_accuracy_metrics:
#     message: "running accuracy metrics on the predictions"
#     input:
#         predictions = rules.aggregate_prediction.output.table,
#         model_truth = "SBIC.dev.scramble.4.csv"
#     output:
#         dataframe = expand("accuracy_values_{version}_{checkpoint}.txt", version=MODEL_VAL, checkpoint = CHECKPOINTS)
#     shell:
#         """
#         python3 accuracy_offNvsY.py \
#             --input-model-truth {input.model_truth} \
#             --input-predictions {input.predictions} \
#             --output {output.dataframe}
#         """

##rule clean:
##    message: "Removing directories: {params}"
##    params:
##        "predictions_1.csv"
##    shell:
##        "rm -rfv {params}"