from snakemake.utils import min_version
min_version("6.0")

MODEL_VAL = ["3"]
rule all:
    input:
        directory(expand('model_{version}', version=MODEL_VAL)),
        expand('predictions_{version}.csv', version=MODEL_VAL),
        #expand("accuracy_values_{version}.csv", version=MODEL_VAL)

# --- TODO: make rule to create data after cleaning up jupyter notebook, add NLG demo, add accuracy script --- #

## rule create_data:
##    message: "creating dataframe in the correct form"
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
        training_csv = "SBIC.trn.scramble.3.5050.csv",
        validation_csv = "SBIC.dev.scramble.3.5050.csv"
    output:
        model = directory(expand('model_{version}', version=MODEL_VAL)),
    shell:
        """
        python3 classifier.py \
            --input-training {input.training_csv} \
            --input-validation {input.validation_csv} \
            --output {output.model}
        """

rule generate_predictions:
    input:
        model = rules.create_model.output.model,
        validation_csv = "SBIC.dev.scramble.3.5050.csv"
    output:
        predictions = expand('predictions_{version}.csv', version=MODEL_VAL)
    shell:
        """
        python3 predictions.py \
            --input-model {input.model} \
            --input-validation {input.validation_csv} \
            --output {output.predictions}
        """

rule get_accuracy_metrics:
    message: "running accuracy metrics on the predictions"
    input:
        predictions = rules.generate_predictions.output.predictions,
        model_truth = "SBIC.dev.scramble.3.5050.csv"
    output:
        dataframe = expand("accuracy_values_{version}.csv", version=MODEL_VAL)
    shell:
        """
        python3 accuracy_metrics.py \
            --input-model-truth {input.model_truth} \
            --input-predictions {input.predictions} \
            --output {output.dataframe}
        """

##rule clean:
##    message: "Removing directories: {params}"
##    params:
##        "predictions_1.csv"
##    shell:
##        "rm -rfv {params}"