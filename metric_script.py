# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

import datasets

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {OffY vs OffN metric},
authors={Sravani Nanduri},
year={2022}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""


# TODO: Define external resources urls if needed
# BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NewMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _compute(self, predictions, references):
        import sys
        import re
        import numpy as np
        accuracy_score = sum((("OffY" in i and "OffY" in j) or ("OffN" in i and "OffN" in j))
                             for i, j in zip(predictions, references)) / len(predictions)

        tokens_input = [re.findall("(?<=\[)[^]]+(?=\])", reference) for reference in references]
        tokens_generated = [re.findall("(?<=\[)[^]]+(?=\])", prediction) for prediction in predictions]
        # compare the arrays elementwise to determine number of tokens correctly preserved, divide by total amount of tokens preserved
        if len(tokens_input) == len(tokens_generated):
            structural_score = sum(x == y for x, y in zip(tokens_input, tokens_generated)) / len(tokens_input)
        else:
            structural_score = 0

        return {"accuracy": (accuracy_score + structural_score) / 2}

# ADD STRUCTURAL ACC

# import re
# import numpy as np
# total_values = []
# # TODO: add in a check to make sure the prediction is correct before adding it to total_values
# for i in range(0, len(merged_df)):
#     # find all tokens between square brackets
#     tokens_input = re.findall("(?<=\[)[^]]+(?=\])", merged_df["output"].tolist()[i])
#     tokens_generated = re.findall("(?<=\[)[^]]+(?=\])", merged_df["generated"].tolist()[i])
#     #compare the arrays elementwise to determine number of tokens correctly preserved, divide by total amount of tokens preserved
#     try:
#         total_values.append((np.array(tokens_input)==np.array(tokens_generated)).sum()/len(tokens_input))
#     #short term solution to deal with arrays that are not the same length: talk in meeting about best way to mitigate
#     except:
#         total_values.append(0);
#
# print(np.mean(total_values))
