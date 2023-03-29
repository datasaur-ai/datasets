"""
copy and past this function into AWS Lambda > Functions > {FUNCTION_NAME}
https://www.notion.so/datasaur/AWS-Comprehend-63149417ef3c4ebb9d84d945d6a5bc31?pvs=4#b692240c9cb84063999db3bc435188da
"""

import json
import boto3


def lambda_handler(event, context):

    try:
        body = json.loads(event["body"])
        final_resp_list = []
        for data in body:
            id_ = data["id"]
            text = data["text"]

            # Get Comprehend client
            comprehend = boto3.client("comprehend")
            # Use detect entities function to get a list of entities
            resp = comprehend.classify_document(
                Text=text,
                EndpointArn="{YOUR_ENDPOINT_ARN}",
            )

            conf_score = [v["Score"] for v in resp["Classes"]]
            conf_score_max = max(conf_score)
            index_max_score = conf_score.index(conf_score_max)

            pred_label = resp["Classes"][index_max_score]["Name"]
            final_resp = {"id": id_, "label": pred_label}
            final_resp_list.append(final_resp)

        return {
            "isBase64Encoded": True,
            "statusCode": "200",
            "headers": {"Content-Type": "text/html"},
            "body": json.dumps(final_resp_list),
        }

    except Exception as e:
        print(e.__class__, " occurred :", e)
        resp = json.dumps({"error": str(e)}, sort_keys=True, indent=4)
        return {"statusCode": "500", "body": resp}
