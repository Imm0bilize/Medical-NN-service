import json


def convert_to_json(data):
    if not data.isinstance(dict):
        # TODO add convert numpy array to json
        prediction = {'prediction': None}
    else:
        prediction = {'prediction': data}
    return prediction
