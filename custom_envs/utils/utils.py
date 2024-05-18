import json
import pathlib
from typing import Iterable

import jsonschema
import numpy as np
from typing import List

PATH_SCHEMAS = pathlib.Path(__file__).parent / ".." / ".." / "configs" / "schemas"
LIST_GLOBAL_SCHEMA_NAME = ["global.json", "learning.json", "model.json"]


def validate_xp_schemas(dict_config: dict) -> None:
    env_name = dict_config["environment"]["name"]
    list_dict_schemas = xp_schema_loading(env_name=env_name)
    validate_jsonschemas(dict_object=dict_config, list_schemas=list_dict_schemas)


def xp_schema_loading(env_name: str) -> List[dict]:
    list_dict_schemas = []
    list_schema_name = LIST_GLOBAL_SCHEMA_NAME + [f"environments/{env_name}.json"]

    for global_schema_name in list_schema_name:
        with open(PATH_SCHEMAS / global_schema_name) as file:
            list_dict_schemas.append(json.load(file))

    return list_dict_schemas


def validate_jsonschemas(dict_object: dict, list_schemas: Iterable[dict]) -> None:
    for dict_schema in list_schemas:
        jsonschema.validate(instance=dict_object, schema=dict_schema)


def complex_to_real_flattened(x: np.array) -> np.array:
    return np.concatenate([x.real, x.imag])


def real_flattened_to_complex(x: np.array) -> np.array:
    x_complex = x[:x.shape[0] // 2] + 1j * x[x.shape[0] // 2:]
    return x_complex
