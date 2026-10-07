"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final
from warnings import catch_warnings, simplefilter

from mdo_framework.core.dependencies import DependencyWalk, walk_dependencies
from mdo_framework.schema import (
    ChoiceVar,
    ConstraintSpec,
    DesignVariable,
    Finding,
    ObjectiveSpec,
    RangeVar,
    StateVar,
    StudySchema,
    ToolSpec,
    ValidationReport,
)

if TYPE_CHECKING:
    from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

INITIAL_TOLERANCE: Final = 1e-9
DEFAULT_COUPLING_GUESS: Final = 0.0

_NUMBER: Final = r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_NAME: Final = r"[A-Za-z_][A-Za-z0-9_]*"
_COMPARISON: Final = re.compile(r"(<=|>=)")
_TERM: Final = re.compile(
    rf"\s*(?P<sign>[+-])?\s*"
    rf"(?:(?P<factor>{_NUMBER})\s*\*\s*(?P<scaled>{_NAME})"
    rf"|(?P<number>{_NUMBER})"
    rf"|(?P<name>{_NAME}))\s*"
)


@dataclass(frozen=True)
class LinearConstraint:
    """Linear inequality over design variables, as ``sum(c * x) <= bound``.

    Attributes:
        expression: Original constraint text.
        coefficients: Coefficient of each referenced variable, none being zero.
        bound: Right-hand side of the normalized inequality.
    """

    expression: str
    coefficients: dict[str, float]
    bound: float


class ParameterConstraintError(ValueError):
    """Raised when a parameter constraint expression is not usable."""


def _invalid_constraint(expression: str, reason: str) -> ParameterConstraintError:
    return ParameterConstraintError(
        f"invalid parameter constraint {expression!r}: {reason}"
    )


def _parse_side(text: str) -> tuple[dict[str, float], float]:
    """Split one side of a comparison into variable coefficients and a constant."""
    coefficients: dict[str, float] = {}
    constant = 0.0
    position = 0
    while True:
        term = _TERM.match(text, position)
        if term is None or (position > 0 and term["sign"] is None):
            raise ValueError(
                "expected terms joined by '+' or '-', each being a number, a name "
                "or '<number>*<name>'"
            )
        sign = -1.0 if term["sign"] == "-" else 1.0
        if term["scaled"] is not None:
            name, factor = term["scaled"], float(term["factor"])
        elif term["name"] is not None:
            name, factor = term["name"], 1.0
        else:
            name, factor = None, float(term["number"])
        if name is None:
            constant += sign * factor
        else:
            coefficients[name] = coefficients.get(name, 0.0) + sign * factor
        position = term.end()
        if position == len(text):
            return coefficients, constant


def _parse_constraint(
    expression: str, design_variables: Mapping[str, DesignVariable]
) -> LinearConstraint:
    parts = _COMPARISON.split(expression)
    if len(parts) != 3:
        raise _invalid_constraint(
            expression, "expected exactly one '<=' or '>=' comparison"
        )
    left_text, operator, right_text = parts
    try:
        left, left_constant = _parse_side(left_text)
        right, right_constant = _parse_side(right_text)
    except ValueError as exc:
        raise _invalid_constraint(expression, str(exc)) from exc

    for name in (*left, *right):
        variable = design_variables.get(name)
        if variable is None:
            raise _invalid_constraint(expression, f"'{name}' is not a design variable")
        if isinstance(variable, ChoiceVar):
            raise _invalid_constraint(
                expression,
                f"'{name}' is a choice variable; parameter constraints only "
                "support range variables",
            )

    merged = dict(left)
    for name, coefficient in right.items():
        merged[name] = merged.get(name, 0.0) - coefficient
    bound = right_constant - left_constant
    if operator == ">=":
        merged = {name: -coefficient for name, coefficient in merged.items()}
        bound = -bound
    coefficients = {name: value for name, value in merged.items() if value != 0.0}

    if not coefficients:
        raise _invalid_constraint(expression, "no variable left to constrain")
    if not all(math.isfinite(value) for value in (*coefficients.values(), bound)):
        raise _invalid_constraint(expression, "numbers must be finite")
    return LinearConstraint(expression, coefficients, bound)


def parse_parameter_constraints(
    expressions: Sequence[str], design_variables: Sequence[DesignVariable]
) -> list[LinearConstraint]:
    """Parse linear constraints written in the Ax grammar.

    Each expression is ``<lhs> <= <rhs>`` or ``<lhs> >= <rhs>`` where a side is
    a sum of numbers, names and ``<number>*<name>`` terms. It is normalized to
    ``sum(coefficient * variable) <= bound``.

    Args:
        expressions: Constraint expressions.
        design_variables: Design variables the expressions may reference.

    Returns:
        One normalized constraint per expression, in order.

    Raises:
        ParameterConstraintError: If an expression cannot be parsed, references
            an unknown variable or a choice variable, or has no variable left.
    """
    by_name = {variable.name: variable for variable in design_variables}
    return [_parse_constraint(expression, by_name) for expression in expressions]


def check_tool_signature(
    tool: ToolSpec, func: Callable[..., Any]
) -> tuple[list[Finding], list[Finding]]:
    """Check that a tool function accepts exactly the graph inputs by keyword.

    Args:
        tool: Tool declaration whose inputs are passed as keyword arguments.
        func: Python callable registered for the tool.

    Returns:
        The error findings and the warning findings.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return [], [
            Finding(
                code="SIGNATURE_UNCHECKED",
                message=f"{tool.name}: signature is not introspectable",
                names=(tool.name,),
            )
        ]
    parameters = list(signature.parameters.values())
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters):
        return [], [
            Finding(
                code="SIGNATURE_UNCHECKED",
                message=f"{tool.name} accepts **kwargs; inputs cannot be checked",
                names=(tool.name,),
            )
        ]

    errors: list[Finding] = []
    try:
        signature.bind(**dict.fromkeys(tool.inputs))
    except TypeError as exc:
        errors.append(
            Finding(
                code="SIGNATURE_MISMATCH",
                message=(
                    f"tool '{tool.name}' is called with the graph inputs "
                    f"{list(tool.inputs)} but its signature {signature} "
                    f"does not accept them: {exc}"
                ),
                names=(tool.name, *tool.inputs),
            )
        )
    warnings = [
        Finding(
            code="DEFAULTED_ARG_UNWIRED",
            message=(
                f"tool '{tool.name}' parameter '{item.name}' has a default "
                "value and is not a graph input; the default will be used"
            ),
            names=(tool.name, item.name),
        )
        for item in parameters
        if item.default is not inspect.Parameter.empty and item.name not in tool.inputs
    ]
    return errors, warnings


def _resolution_errors(walk: DependencyWalk, has_targets: bool) -> list[Finding]:
    errors = [
        Finding(
            code="UNKNOWN_OUTPUT",
            message=f"'{name}' is not a declared variable",
            names=(name,),
        )
        for name in walk.unknown_targets
    ]
    errors.extend(
        Finding(
            code="NOT_PRODUCED",
            message=f"'{name}' is not produced by any tool",
            names=(name,),
        )
        for name in walk.unproduced_targets
    )
    errors.extend(
        Finding(
            code="UNPRODUCED_STATE",
            message=(
                f"state variable '{name}' is an input of a required tool "
                "but no tool produces it"
            ),
            names=(name,),
        )
        for name in walk.unproduced_states
    )
    resolved = not (walk.unknown_targets or walk.unproduced_targets)
    if has_targets and resolved and not walk.design_variables:
        errors.append(
            Finding(
                code="NO_DESIGN_VARIABLES",
                message="the requested outputs do not depend on any design variable",
            )
        )
    return errors


def _registry_findings(
    schema: StudySchema, registry: Mapping[str, Callable[..., Any]]
) -> tuple[list[Finding], list[Finding]]:
    errors = [
        Finding(
            code="UNREGISTERED_TOOL",
            message=f"tool '{tool.name}' has no registered function",
            names=(tool.name,),
        )
        for tool in schema.tools
        if tool.name not in registry
    ]
    warnings: list[Finding] = []
    for tool in schema.tools:
        if tool.name in registry:
            tool_errors, tool_warnings = check_tool_signature(tool, registry[tool.name])
            errors.extend(tool_errors)
            warnings.extend(tool_warnings)
    return errors, warnings


def _parse_each(
    expressions: Sequence[str], design_variables: Sequence[DesignVariable]
) -> tuple[list[LinearConstraint], list[Finding]]:
    parsed: list[LinearConstraint] = []
    errors: list[Finding] = []
    for expression in expressions:
        try:
            parsed.extend(parse_parameter_constraints([expression], design_variables))
        except ParameterConstraintError as exc:
            errors.append(
                Finding(
                    code="PARAMETER_CONSTRAINT_INVALID",
                    message=str(exc),
                    names=(expression,),
                )
            )
    return parsed, errors


def _initial_violations(
    constraints: Sequence[LinearConstraint], design_variables: Sequence[DesignVariable]
) -> list[Finding]:
    initials = {
        variable.name: variable.initial
        for variable in design_variables
        if isinstance(variable, RangeVar) and variable.initial is not None
    }
    findings: list[Finding] = []
    for constraint in constraints:
        if not all(name in initials for name in constraint.coefficients):
            continue
        value = sum(
            coefficient * initials[name]
            for name, coefficient in constraint.coefficients.items()
        )
        limit = constraint.bound + INITIAL_TOLERANCE * max(1.0, abs(constraint.bound))
        if value > limit:
            findings.append(
                Finding(
                    code="INITIAL_OUT_OF_SPACE",
                    message=(
                        f"initial values violate '{constraint.expression}': "
                        f"{value:g} exceeds the bound {constraint.bound:g}"
                    ),
                    names=(constraint.expression, *constraint.coefficients),
                )
            )
    return findings


def _gemseo_variable_arguments(variable: DesignVariable) -> dict[str, Any]:
    if isinstance(variable, RangeVar):
        return {
            "lower_bound": variable.lower,
            "upper_bound": variable.upper,
            "type_": "integer" if variable.value_type == "int" else "float",
            "value": variable.initial,
        }
    return {
        "lower_bound": 0,
        "upper_bound": len(variable.choices) - 1,
        "type_": "integer",
        "value": (
            None
            if variable.initial is None
            else variable.choices.index(variable.initial)
        ),
    }


def _design_space_errors(design_variables: Sequence[DesignVariable]) -> list[Finding]:
    # GEMSEO is imported lazily: its import is slow and emits third-party warnings.
    with catch_warnings():
        simplefilter("ignore")
        from gemseo.algos.design_space import DesignSpace

        try:
            design_space = DesignSpace()
        except Exception as exc:
            return [
                Finding(
                    code="DESIGN_SPACE_INVALID",
                    message=f"GEMSEO cannot create a design space: {exc}",
                    names=tuple(variable.name for variable in design_variables),
                )
            ]
        errors: list[Finding] = []
        for variable in design_variables:
            try:
                design_space.add_variable(
                    variable.name, **_gemseo_variable_arguments(variable)
                )
            except Exception as exc:
                errors.append(
                    Finding(
                        code="DESIGN_SPACE_INVALID",
                        message=(
                            f"GEMSEO rejects design variable '{variable.name}': {exc}"
                        ),
                        names=(variable.name,),
                    )
                )
    return errors


def _ax_parameters(
    design_variables: Sequence[DesignVariable],
) -> list["RangeParameterConfig | ChoiceParameterConfig"]:
    from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

    parameters: list[RangeParameterConfig | ChoiceParameterConfig] = []
    for variable in design_variables:
        if isinstance(variable, RangeVar):
            cast = int if variable.value_type == "int" else float
            parameters.append(
                RangeParameterConfig(
                    name=variable.name,
                    bounds=(cast(variable.lower), cast(variable.upper)),
                    parameter_type=variable.value_type,
                )
            )
        else:
            parameters.append(
                ChoiceParameterConfig(
                    name=variable.name,
                    values=list(variable.choices),
                    parameter_type=variable.value_type,
                )
            )
    return parameters


def _ax_constraint_errors(
    design_variables: Sequence[DesignVariable], expressions: Sequence[str]
) -> list[Finding]:
    try:
        # Ax is imported lazily: its import is slow and emits third-party warnings.
        with catch_warnings():
            simplefilter("ignore")
            from ax.api.client import Client

            Client().configure_experiment(
                name="validation",
                parameters=_ax_parameters(design_variables),
                parameter_constraints=list(expressions),
            )
    except Exception as exc:
        return [
            Finding(
                code="PARAMETER_CONSTRAINT_INVALID",
                message=f"Ax rejects the parameter constraints: {exc}",
                names=tuple(expressions),
            )
        ]
    return []


def _search_space_errors(
    design_variables: Sequence[DesignVariable], expressions: Sequence[str]
) -> list[Finding]:
    """Check the parameter constraints, then dry-build the Ax and GEMSEO spaces."""
    constraints, errors = _parse_each(expressions, design_variables)
    can_dry_build = bool(design_variables) and not errors
    if can_dry_build and expressions:
        errors.extend(_ax_constraint_errors(design_variables, expressions))
    errors.extend(_initial_violations(constraints, design_variables))
    if can_dry_build:
        errors.extend(_design_space_errors(design_variables))
    return errors


def _coupling_warnings(schema: StudySchema, walk: DependencyWalk) -> list[Finding]:
    return [
        Finding(
            code="COUPLING_DEFAULT_GUESS",
            message=(
                f"coupling variable '{name}' has no initial_guess; "
                f"{DEFAULT_COUPLING_GUESS} will be used"
            ),
            names=(name,),
        )
        for name in walk.couplings
        if isinstance(variable := schema.variable(name), StateVar)
        and variable.initial_guess is None
    ]


def _unused_warnings(
    schema: StudySchema, walk: DependencyWalk, has_targets: bool
) -> list[Finding]:
    referenced = {
        name for tool in schema.tools for name in (*tool.inputs, *tool.outputs)
    }
    warnings = [
        Finding(
            code="UNUSED_VARIABLE",
            message=f"variable '{variable.name}' is not used by any tool",
            names=(variable.name,),
        )
        for variable in schema.variables
        if variable.name not in referenced
    ]
    resolved = not (walk.unknown_targets or walk.unproduced_targets)
    if has_targets and resolved:
        warnings.extend(
            Finding(
                code="UNUSED_TOOL",
                message=(
                    f"tool '{tool.name}' is not needed by the objectives "
                    "and constraints"
                ),
                names=(tool.name,),
            )
            for tool in schema.tools
            if tool.name not in walk.tools
        )
    return warnings


def _partial_initial_warnings(
    design_variables: Sequence[DesignVariable],
) -> list[Finding]:
    missing = [
        variable.name for variable in design_variables if variable.initial is None
    ]
    if not missing or len(missing) == len(design_variables):
        return []
    return [
        Finding(
            code="PARTIAL_INITIAL",
            message=(
                f"design variables {missing} have no initial value while others do; "
                "x0 will not be auto-enabled"
            ),
            names=tuple(missing),
        )
    ]


def validate_study(
    schema: StudySchema,
    *,
    objectives: Sequence[ObjectiveSpec],
    constraints: Sequence[ConstraintSpec] = (),
    parameter_constraints: Sequence[str] = (),
    registry: Mapping[str, Callable[..., Any]] | None = None,
) -> ValidationReport:
    """Check that a study can run, without calling any tool.

    Every applicable finding is reported; nothing stops at the first error.

    Args:
        schema: Study to validate.
        objectives: Objectives, whose names are the first targets.
        constraints: Constraints, whose names follow the objectives as targets.
        parameter_constraints: Linear constraints on the design variables, in
            the Ax grammar.
        registry: Optional mapping of tool name to Python function. When given,
            every schema tool must be registered with a compatible signature.

    Returns:
        Report listing the errors and warnings.
    """
    targets = [item.name for item in (*objectives, *constraints)]
    walk = walk_dependencies(schema, targets)
    has_targets = bool(targets)

    errors = _resolution_errors(walk, has_targets)
    signature_warnings: list[Finding] = []
    if registry is not None:
        registry_errors, signature_warnings = _registry_findings(schema, registry)
        errors.extend(registry_errors)
    errors.extend(_search_space_errors(walk.design_variables, parameter_constraints))

    warnings = [
        *_coupling_warnings(schema, walk),
        *_unused_warnings(schema, walk, has_targets),
        *signature_warnings,
        *_partial_initial_warnings(walk.design_variables),
    ]
    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))
