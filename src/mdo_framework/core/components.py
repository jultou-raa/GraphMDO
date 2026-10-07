"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from gemseo.core.discipline import Discipline
from gemseo.core.discipline.base_discipline import CacheType
from gemseo.typing import StrKeyMapping

from mdo_framework.core.errors import ToolExecutionError, ToolOutputError
from mdo_framework.optimization.parameter_codec import (
    ParameterDefinition,
    index_to_value,
)
from mdo_framework.schema import argument_collisions


def to_tool_value(spec: ParameterDefinition | None, raw: Any) -> Any:
    """Converts a GEMSEO input value into the value declared for the tool.

    Choice variables travel through GEMSEO as indices and are decoded to the
    declared choice; integer variables are delivered as ``int``. Vectors and
    variables without a spec are passed through (scalars unwrapped).
    """
    if isinstance(raw, np.ndarray) and raw.size != 1:
        return raw
    value = raw.item() if isinstance(raw, (np.ndarray, np.generic)) else raw
    if spec is None:
        return value
    if spec.get("type") == "choice":
        return index_to_value(spec, value)
    if spec.get("value_type") == "int" and not isinstance(value, bool):
        return int(round(float(value)))
    return value


class ToolComponent(Discipline):
    """GEMSEO discipline that wraps a Python function.

    The function is called with keyword arguments only, one per input. It returns
    a dict keyed by output name, or a bare value when the tool has a single
    output; a list or an array is a vector value of a single output, a tuple is
    never accepted. Anything else, a non-finite value included, is a
    ``ToolOutputError``; an exception raised by the function becomes a
    ``ToolExecutionError``. Two inputs cannot be passed as the same argument.
    Jacobians are approximated by finite differences.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        inputs: Sequence[str],
        outputs: Sequence[str],
        *,
        specs: Mapping[str, ParameterDefinition] | None = None,
        defaults: Mapping[str, Any] | None = None,
        deterministic: bool = True,
        arg_map: Mapping[str, str] | None = None,
    ) -> None:
        """Initializes the tool component.

        Args:
            name: The name of the discipline.
            func: The Python callable executing the tool logic.
            inputs: Input variable names.
            outputs: Output variable names.
            specs: Parameter definitions per input name, used to decode choice
                indices and integers before calling ``func``.
            defaults: Value of the inputs that may be omitted when executing, per
                input name. An input without a default is required.
            deterministic: Whether the same inputs always give the same outputs.
                If false, the discipline is never cached.
            arg_map: Python argument name per input name, for the inputs whose
                argument is named differently.
        """
        super().__init__(name=name)
        self.func = func
        self._input_names = tuple(inputs)
        self._output_names = tuple(outputs)
        self._specs = dict(specs or {})
        self._arg_map = dict(arg_map or {})
        collisions = argument_collisions(self._input_names, self._arg_map)
        if collisions:
            clashes = "; ".join(
                f"'{argument}' would receive {names}"
                for argument, names in collisions.items()
            )
            raise ValueError(
                f"Tool '{name}': several inputs are passed as the same argument: "
                f"{clashes}; map each input to its own argument"
            )
        self.input_grammar.update_from_names(self._input_names)
        self.output_grammar.update_from_names(self._output_names)
        self.default_input_data = {
            input_name: np.atleast_1d(value)
            for input_name, value in (defaults or {}).items()
        }
        self.set_jacobian_approximation()
        if not deterministic:
            self.set_cache(CacheType.NONE)

    def _run(self, input_data: StrKeyMapping) -> dict[str, np.ndarray]:
        kwargs = {
            self._arg_map.get(name, name): to_tool_value(
                self._specs.get(name), input_data[name]
            )
            for name in self._input_names
        }
        try:
            result = self.func(**kwargs)
        except MemoryError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"{type(exc).__name__}: {exc}", tool=self.name
            ) from exc
        return self._normalise_outputs(result)

    def _normalise_outputs(self, result: Any) -> dict[str, np.ndarray]:
        if result is None:
            raise self._output_error(
                f"returned None; expected values for {list(self._output_names)}"
            )
        if isinstance(result, Mapping):
            values = self._values_by_name(result)
        elif isinstance(result, tuple):
            raise self._output_error(
                "returned a tuple, which is never matched to the outputs "
                f"{list(self._output_names)}; return a dict keyed by output name, "
                "or a single value for a tool with one output"
            )
        elif len(self._output_names) == 1:
            values = {self._output_names[0]: result}
        else:
            raise self._output_error(
                f"returned a {type(result).__name__} for the outputs "
                f"{list(self._output_names)}; return a dict keyed by output name"
            )

        outputs: dict[str, np.ndarray] = {}
        for output_name, value in values.items():
            try:
                outputs[output_name] = np.atleast_1d(np.asarray(value, dtype=float))
            except (TypeError, ValueError) as exc:
                raise self._output_error(
                    f"output '{output_name}' is not numeric: {exc}"
                ) from exc
        non_finite = [
            output_name
            for output_name, array in outputs.items()
            if not np.all(np.isfinite(array))
        ]
        if non_finite:
            raise self._output_error(
                f"non-finite value (NaN or infinity) in the outputs {non_finite}"
            )
        return outputs

    def _values_by_name(self, result: Mapping[str, Any]) -> dict[str, Any]:
        missing = [name for name in self._output_names if name not in result]
        unexpected = [name for name in result if name not in self._output_names]
        if missing or unexpected:
            raise self._output_error(
                "the returned dict does not match the declared outputs "
                f"{list(self._output_names)}: missing {missing}, "
                f"unexpected {unexpected}"
            )
        return {name: result[name] for name in self._output_names}

    def _output_error(self, message: str) -> ToolOutputError:
        return ToolOutputError(message, tool=self.name)
