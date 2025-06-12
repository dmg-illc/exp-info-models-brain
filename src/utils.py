import json

def open_json(path):
    with open(path, 'r') as file:
        loaded_dict = json.load(file)
    return loaded_dict

def dict_to_json(dict_to_save, path):
    with open(path, 'w') as file:
        json.dump(dict_to_save, file)