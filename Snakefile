from snakemake.utils import min_version
min_version("6.0")

MODEL_VAL = ["1"]
rule all:
    input:
        expand('model_{version}', version=MODEL_VAL),
        expand('/predictions_{version}.csv', version=MODEL_VAL)

""" TODO: make rule to create data after cleaning up jupyter notebook, add NLG demo, add accuracy script """
rule create_model:
    input:
        training_csv = "SBIC.trn.1.csv",
        validation_csv = "SBIC.dev.1.csv"
    output:
        model = expand('model_{version}', version=MODEL_VAL),
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
        validation_csv = "SBIC.dev.1.csv"
    output:
        predictions = expand('/predictions_{version}.csv', version=MODEL_VAL)
    shell:
        """
        python3 predictions.py \
            --input-model {input.model} \
            --input-validation {input.validation_csv} \
            --output {output.predictions}
        """

"""
rule get_accuracy_metrics:
message: "running PCA before removing any low quality strains for supplemental figure bases missing vs pc1"
input:
    alignment = rules.extract_sequences.output.sequences
output:
    dataframe = "results/embed_pca_before.csv"
params:
    components = 10
conda: "../cartography.yml"
shell:
"""
"""
"""
rule clean:
    message: "Removing directories: {params}"
    params:
        "model_1"
        "predictions_1.csv"
        "predictions_1.xslx"
    shell:
        "rm -rfv {params}"