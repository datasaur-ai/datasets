# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

input_dict_list = [{"id": 0, "text": "data_sample"}]
input_sample = StandardPythonParameterType(input_dict_list)

result_sample = [{"id": 0, "text": "Your text here."}]
output_sample = StandardPythonParameterType(result_sample)

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_v2')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions(
        {'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

# Sample API request body
# [
#   { "id": 0, "text": "I feel good." },
#   { "id": 1, "text": "I don't like it." }
# ]

# Sample API response
# [
#   { "id": 0, "label": "POSITIVE" },
#   { "id": 1, "label": "NEGATIVE" }
# ]


def input_converter(inputs):
    text_list = []

    for data_dict in inputs:
        text_list.append(data_dict['text'])

    df = pd.DataFrame({"Text": text_list})

    return df


def result_converter(inputs, results):
    data_list = []

    for data_dict, text in zip(inputs, results):
        id = data_dict['id']

        current_dict = {
            "id": id,
            "label": text
        }

        data_list.append(current_dict)

    return data_list


def run(inputs):
    inputs = json.loads(inputs)
    data = input_converter(inputs)

    results = model.predict(data)

    if isinstance(results, pd.DataFrame):
        results = results.values

    results_list = results.tolist()

    return result_converter(inputs, results_list)
