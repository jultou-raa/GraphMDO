"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from mdo_framework.core.dependencies import walk_dependencies
from mdo_framework.optimization.parameter_codec import ParameterDefinition
from mdo_framework.schema import (
    ChoiceVar,
    DesignVariable,
    FixedParam,
    RangeVar,
    StudySchema,
    StudyValidationError,
    ValidationReport,
)
from mdo_framework.validation import dependency_findings


@dataclass(frozen=True)
class ResolvedInputs:
    """Inputs and tools required to evaluate a set of target outputs.

    All tuples follow the declaration order of the schema.

    Attributes:
        design_variables: Bounded or choice inputs the optimizer may vary.
        fixed_parameters: Valued inputs that keep their value.
        tools: Names of the tools to execute.
    """

    design_variables: tuple[DesignVariable, ...]
    fixed_parameters: tuple[FixedParam, ...]
    tools: tuple[str, ...]


class TopologicalAnalyzer:
    """Resolves which inputs and tools a set of target outputs depends on."""

    def __init__(self, schema: StudySchema):
        """Initializes the analyzer with the study schema.

        Args:
            schema: Typed description of the variables and tools.

        """
        self.schema = schema

    def resolve_dependencies(self, target_outputs: Sequence[str]) -> ResolvedInputs:
        """Resolve all dependencies needed to compute ``target_outputs``.

        Args:
            target_outputs: Names of the variables to compute.

        Returns:
            The design variables, fixed parameters and tools that the targets
            depend on, in schema order.

        Raises:
            StudyValidationError: If a target is unknown or not produced by a
                tool, or a required tool consumes a state no tool produces.
                Every finding is reported in the same error.
        """
        walk = walk_dependencies(self.schema, target_outputs)
        errors = dependency_findings(walk)
        if errors:
            raise StudyValidationError(ValidationReport(errors=tuple(errors)))
        return ResolvedInputs(
            design_variables=walk.design_variables,
            fixed_parameters=walk.fixed_parameters,
            tools=walk.tools,
        )

    def extract_parameters(
        self, design_variables: Sequence[DesignVariable]
    ) -> list[ParameterDefinition]:
        """Format design variables into optimizer-ready parameter definitions."""
        return [to_parameter_definition(variable) for variable in design_variables]


def to_parameter_definition(variable: DesignVariable) -> ParameterDefinition:
    """Converts a design variable into a parameter definition.

    The definition is shared by the optimizer (Ax/GEMSEO design space) and the
    tool boundary, so both decode choice indices and integers identically.
    """
    if isinstance(variable, RangeVar):
        return {
            "name": variable.name,
            "type": "range",
            "bounds": [variable.lower, variable.upper],
            "value_type": variable.value_type,
        }
    return {
        "name": variable.name,
        "type": "choice",
        "values": list(variable.choices),
        "value_type": variable.value_type,
    }


def to_fixed_parameter_definition(variable: FixedParam) -> ParameterDefinition | None:
    """Converts a fixed parameter into a parameter definition, if it needs one.

    GEMSEO only carries numbers, so a ``str`` or ``bool`` value travels as the
    index of a one-value choice and an ``int`` value is pinned by a degenerate
    integer range. The tool then receives the declared value and type. Floats
    need no definition.
    """
    value = variable.value
    if isinstance(value, bool):
        value_type = "bool"
    elif isinstance(value, str):
        value_type = "str"
    elif isinstance(value, int):
        return {
            "name": variable.name,
            "type": "range",
            "bounds": [value, value],
            "value_type": "int",
        }
    else:
        return None
    return {
        "name": variable.name,
        "type": "choice",
        "values": [value],
        "value_type": value_type,
    }


def build_variable_specs(schema: StudySchema) -> dict[str, ParameterDefinition]:
    """Maps design variables and non-float fixed parameters to their definitions.

    Entries follow the declaration order of the schema.
    """
    specs: dict[str, ParameterDefinition] = {}
    for variable in schema.variables:
        if isinstance(variable, RangeVar | ChoiceVar):
            specs[variable.name] = to_parameter_definition(variable)
        elif isinstance(variable, FixedParam):
            definition = to_fixed_parameter_definition(variable)
            if definition is not None:
                specs[variable.name] = definition
    return specs
