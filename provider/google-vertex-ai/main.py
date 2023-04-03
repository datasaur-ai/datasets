import json

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_text_entity_extraction_sample(
    project: str,
    endpoint_id: str,
    content: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema
    instance = predict.instance.TextExtractionPredictionInstance(
        content=content,
    ).to_value()
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)

    # See gs://google-cloud-aiplatform/schema/predict/prediction/text_extraction_1.0.0.yaml for the format of the predictions.
    prediction_result = {}
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))
        prediction_result = dict(prediction)

    return prediction_result


def preprocess_data(data):
    project_id = data["id"]
    documents_id = data["documents"][0]["id"]

    dict_text = {}
    for sentence in data["documents"][0]["sentences"]:
        text = sentence["text"]
        sentence_id = sentence["id"]
        dict_text[sentence_id] = text

    return project_id, documents_id, dict_text


def postprocess_data(project_id, documents_id, labels_list):
    output = {}
    output["id"] = project_id
    output["documents"] = [{"id": documents_id, "labels": labels_list}]
    return output


def run_prediction(request):
    data_input = request.get_json()

    print(data_input)

    project_id, documents_id, dict_text = preprocess_data(data_input)

    labels_list = []
    for sentence_id, text in dict_text.items():

        output = predict_text_entity_extraction_sample(
            project="816942540773",
            endpoint_id="3819839734335668224",
            location="us-central1",
            content=text,
        )

        labels = output["displayNames"]
        confidences = output["confidences"]
        start_chars = output["textSegmentStartOffsets"]
        end_chars = output["textSegmentEndOffsets"]

        entities = []
        for start_char, end_char, label in zip(start_chars, end_chars, labels):
            entity = {}
            entity["start_char"] = int(start_char)
            entity["end_char"] = int(end_char)
            entity["label"] = label
            entity["layer"] = 0
            entities.append(entity)

        labels_dict = {"id": sentence_id, "entities": entities}

        labels_list.append(labels_dict)

    predictions = postprocess_data(project_id, documents_id, labels_list)

    return json.dumps(predictions)
