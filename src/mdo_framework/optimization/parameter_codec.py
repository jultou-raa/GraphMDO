"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

ScalarValue: TypeAlias = bool | int | float | str
ParameterDefinition: TypeAlias = Mapping[str, Any]


class ParameterCodecError(ValueError):
    """Raised when parameter encoding or decoding fails."""


class ParameterDefinitionError(ParameterCodecError):
    """Raised when a parameter definition is invalid."""


class ParameterValueError(ParameterCodecError):
    """Raised when a parameter value cannot be encoded or decoded."""


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_parameter_lookup(
    parameters: Sequence[ParameterDefinition] | None,
) -> dict[str, ParameterDefinition]:
    return {parameter["name"]: parameter for parameter in parameters or []}


def _get_choice_values(parameter: ParameterDefinition) -> list[Any]:
    choices = parameter.get("values", [])
    if not choices:
        raise ParameterDefinitionError(
            f"Choice parameter {parameter['name']} requires at least one value."
        )
    return choices


def _choice_values_match(choice_value: Any, candidate: Any) -> bool:
    if isinstance(choice_value, bool) or isinstance(candidate, bool):
        return type(choice_value) is type(candidate) and choice_value == candidate
    return choice_value == candidate


def _find_choice_index(choices: list[Any], value: Any) -> int | None:
    for index, choice in enumerate(choices):
        if _choice_values_match(choice, value):
            return index
    return None


def encode_parameter_value(
    parameter: ParameterDefinition | None,
    raw_value: Any,
) -> float:
    value = coerce_scalar(raw_value)
    if parameter is None:
        return float(value)

    if parameter["type"] == "choice":
        choices = _get_choice_values(parameter)
        if len(choices) == 1:
            return 0.0

        choice_index = _find_choice_index(choices, value)
        if choice_index is not None:
            return float(choice_index)

        if isinstance(value, bool):
            raise ParameterValueError(
                f"Cannot encode choice parameter {parameter['name']} from value {value!r}."
            )

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ParameterValueError(
                f"Cannot encode choice parameter {parameter['name']} from value {value!r}."
            ) from exc

    if parameter.get("value_type") == "int" and not isinstance(value, bool):
        return float(int(round(float(value))))

    return float(value)


def decode_parameter_value(
    parameter: ParameterDefinition | None,
    raw_value: Any,
) -> ScalarValue | Any:
    value = coerce_scalar(raw_value)
    if parameter is None:
        return value

    if parameter["type"] == "choice":
        choices = _get_choice_values(parameter)
        choice_index = _find_choice_index(choices, value)
        if choice_index is not None:
            return choices[choice_index]

        if isinstance(value, bool):
            raise ParameterValueError(
                f"Cannot decode choice parameter {parameter['name']} from value {value!r}."
            )

        try:
            index = int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise ParameterValueError(
                f"Cannot decode choice parameter {parameter['name']} from value {value!r}."
            ) from exc
        if not 0 <= index < len(choices):
            raise ParameterValueError(
                f"Choice index {index} is out of bounds for parameter {parameter['name']}."
            )
        return choices[index]

    if parameter.get("value_type") == "int" and not isinstance(value, bool):
        return int(round(float(value)))

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return float(value)

    return value
