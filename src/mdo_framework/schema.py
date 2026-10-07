"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import hashlib
import json
import keyword
from typing import Annotated, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    ValidationError,
    computed_field,
    model_validator,
)
from pydantic_core import PydanticCustomError

MAX_NAME_LENGTH: Final = 50
NAME_PATTERN: Final = r"^[A-Za-z_][A-Za-z0-9_]*$"


def _reject_python_keyword(name: str) -> str:
    if keyword.iskeyword(name):
        raise ValueError(f"'{name}' is a Python keyword and cannot be a tool argument")
    return name


Name = Annotated[
    str,
    StringConstraints(pattern=NAME_PATTERN, max_length=MAX_NAME_LENGTH),
    AfterValidator(_reject_python_keyword),
]
"""Identifier usable as a keyword argument: pattern, length and keyword checked."""

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
"""Finite float: rejects NaN, infinities, ``bool`` and ``str`` (``int`` is widened)."""

Scalar = (
    StrictBool
    | StrictInt
    | Annotated[StrictFloat, Field(allow_inf_nan=False)]
    | StrictStr
)
"""Exact-typed scalar: ``bool``, ``int``, finite ``float`` or ``str``."""

_VALUE_TYPE_NAMES: Final = {bool: "bool", int: "int", float: "float", str: "str"}


class _Strict(BaseModel):
    """Base class: unknown keys are rejected and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _raise_if_any(violations: list[str]) -> None:
    if violations:
        raise ValueError("; ".join(violations))


class RangeVar(_Strict):
    """Continuous or integer design variable bounded by ``[lower, upper]``.

    Attributes:
        kind: Discriminator, always ``"range"``.
        name: Variable name.
        lower: Inclusive lower bound.
        upper: Inclusive upper bound, strictly above ``lower``.
        value_type: ``"int"`` for integral values, ``"float"`` otherwise.
        initial: Optional starting point within the bounds.
        units: Optional free-form unit label.
    """

    kind: Literal["range"] = "range"
    name: Name
    lower: Finite
    upper: Finite
    value_type: Literal["float", "int"] = "float"
    initial: Finite | None = None
    units: str | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> "RangeVar":
        violations: list[str] = []
        if not self.lower < self.upper:
            violations.append(
                f"lower ({self.lower}) must be strictly below upper ({self.upper})"
            )
        if self.value_type == "int":
            for label, value in (
                ("lower", self.lower),
                ("upper", self.upper),
                ("initial", self.initial),
            ):
                if value is not None and not value.is_integer():
                    violations.append(
                        f"{label} ({value}) must be integral when value_type is 'int'"
                    )
        if self.initial is not None and not self.lower <= self.initial <= self.upper:
            violations.append(
                f"initial ({self.initial}) must lie within [{self.lower}, {self.upper}]"
            )
        _raise_if_any(violations)
        return self


class ChoiceVar(_Strict):
    """Categorical design variable taking one of several homogeneous values.

    Attributes:
        kind: Discriminator, always ``"choice"``.
        name: Variable name.
        choices: At least two unique values sharing one Python type.
        initial: Optional starting value, one of ``choices``.
        units: Optional free-form unit label.
    """

    kind: Literal["choice"] = "choice"
    name: Name
    choices: list[Scalar]
    initial: Scalar | None = None
    units: str | None = None

    @model_validator(mode="after")
    def _check_choices(self) -> "ChoiceVar":
        violations: list[str] = []
        if len(self.choices) < 2:
            violations.append(
                "choices must contain at least 2 values; declare a single value "
                "as a fixed parameter"
            )
        if len({type(choice) for choice in self.choices}) > 1:
            violations.append("choices must all be of the same type")
        if len({(type(choice), choice) for choice in self.choices}) < len(self.choices):
            violations.append("choices must be unique")
        if self.initial is not None and not any(
            type(choice) is type(self.initial) and choice == self.initial
            for choice in self.choices
        ):
            violations.append(
                f"initial ({self.initial!r}) must be one of the choices "
                "with the same type"
            )
        _raise_if_any(violations)
        return self

    @property
    def value_type(self) -> Literal["bool", "int", "float", "str"]:
        """Type name of the choices, as expected by the parameter codec."""
        return _VALUE_TYPE_NAMES[type(self.choices[0])]


class FixedParam(_Strict):
    """Constant input passed unchanged to the tools that declare it.

    Attributes:
        kind: Discriminator, always ``"fixed"``.
        name: Variable name.
        value: The constant value.
        units: Optional free-form unit label.
    """

    kind: Literal["fixed"] = "fixed"
    name: Name
    value: Scalar
    units: str | None = None


class StateVar(_Strict):
    """Variable produced by a tool (an output or a coupling variable).

    Attributes:
        kind: Discriminator, always ``"state"``.
        name: Variable name.
        initial_guess: Optional scalar or non-empty list used to seed couplings.
        units: Optional free-form unit label.
    """

    kind: Literal["state"] = "state"
    name: Name
    initial_guess: Finite | Annotated[list[Finite], Field(min_length=1)] | None = None
    units: str | None = None


Variable = Annotated[
    RangeVar | ChoiceVar | FixedParam | StateVar, Field(discriminator="kind")
]
DesignVariable = RangeVar | ChoiceVar


class ToolSpec(_Strict):
    """Tool (discipline) with its input and output variable names.

    Attributes:
        name: Tool name.
        inputs: Unique names of the variables the tool reads.
        outputs: Unique names of the variables the tool produces.
        fidelity: Fidelity level label.
    """

    name: Name
    inputs: list[Name] = []
    outputs: list[Name] = []
    fidelity: Name = "high"

    @model_validator(mode="after")
    def _check_ports(self) -> "ToolSpec":
        violations: list[str] = []
        if len(set(self.inputs)) < len(self.inputs):
            violations.append("inputs must be unique")
        if len(set(self.outputs)) < len(self.outputs):
            violations.append("outputs must be unique")
        shared = sorted(set(self.inputs) & set(self.outputs))
        if shared:
            violations.append(
                f"names cannot be both input and output of the same tool: {shared}"
            )
        _raise_if_any(violations)
        return self


class ObjectiveSpec(_Strict):
    """Optimization objective.

    Attributes:
        name: Name of the produced variable to optimize.
        minimize: ``True`` to minimize, ``False`` to maximize.
        threshold: Optional reference point for multi-objective runs.
    """

    name: Name
    minimize: bool = True
    threshold: Finite | None = None


class ConstraintSpec(_Strict):
    """Inequality constraint on a produced variable.

    Attributes:
        name: Name of the produced variable to constrain.
        bound: Constraint bound.
        op: ``"<="`` for an upper bound, ``">="`` for a lower bound.
    """

    name: Name
    bound: Finite
    op: Literal["<=", ">="] = "<="


class Finding(_Strict):
    """Single validation issue.

    Attributes:
        code: Stable machine-readable code.
        message: Human-readable description.
        names: Variable, tool or field names involved.
    """

    code: str
    message: str
    names: tuple[str, ...] = ()


class ValidationReport(_Strict):
    """Outcome of validating a study.

    Attributes:
        errors: Findings that make the study unusable.
        warnings: Findings that do not block the study.
        valid: Computed flag, ``True`` when there are no errors.
    """

    errors: tuple[Finding, ...] = ()
    warnings: tuple[Finding, ...] = ()

    @computed_field
    @property
    def valid(self) -> bool:
        """Whether the report contains no error."""
        return not self.errors


class StudyValidationError(ValueError):
    """Raised when a study fails validation.

    Attributes:
        report: The full validation report.
    """

    def __init__(self, report: ValidationReport) -> None:
        """Build the error from a report.

        Args:
            report: Report whose errors are listed in the message.
        """
        self.report = report
        super().__init__(
            "; ".join(f"{finding.code}: {finding.message}" for finding in report.errors)
        )


class StudySchema(_Strict):
    """Typed, shared description of a study: variables and tools.

    Variable order is significant: design variables keep their declaration order.

    Attributes:
        schema_version: Schema format version.
        variables: Declared variables.
        tools: Declared tools.
    """

    schema_version: Literal["1"] = "1"
    variables: list[Variable] = []
    tools: list[ToolSpec] = []

    @model_validator(mode="after")
    def _check_structure(self) -> "StudySchema":
        findings = [
            *self._duplicate_name_findings(),
            *self._undeclared_ref_findings(),
            *self._producer_findings(),
        ]
        if findings:
            raise PydanticCustomError(
                "study_structure",
                "; ".join(f"{finding.code}: {finding.message}" for finding in findings),
                {"findings": [finding.model_dump() for finding in findings]},
            )
        return self

    def _duplicate_name_findings(self) -> list[Finding]:
        findings: list[Finding] = []
        for label, names in (
            ("variable", [variable.name for variable in self.variables]),
            ("tool", [tool.name for tool in self.tools]),
        ):
            for name in dict.fromkeys(names):
                if names.count(name) > 1:
                    findings.append(
                        Finding(
                            code="DUPLICATE_NAME",
                            message=f"{label} '{name}' is declared more than once",
                            names=(name,),
                        )
                    )
        return findings

    def _undeclared_ref_findings(self) -> list[Finding]:
        declared = {variable.name for variable in self.variables}
        findings: list[Finding] = []
        for tool in self.tools:
            for role, names in (("input", tool.inputs), ("output", tool.outputs)):
                findings.extend(
                    Finding(
                        code="UNDECLARED_REF",
                        message=(
                            f"tool '{tool.name}' {role} '{name}' "
                            "is not a declared variable"
                        ),
                        names=(name, tool.name),
                    )
                    for name in names
                    if name not in declared
                )
        return findings

    def _producer_findings(self) -> list[Finding]:
        kinds: dict[str, str] = {}
        for variable in self.variables:
            kinds.setdefault(variable.name, variable.kind)
        producing_tools: dict[str, list[str]] = {}
        for tool in self.tools:
            for name in tool.outputs:
                if name in kinds:
                    producing_tools.setdefault(name, []).append(tool.name)

        findings: list[Finding] = []
        for name, tools in producing_tools.items():
            if len(tools) > 1:
                findings.append(
                    Finding(
                        code="DUPLICATE_PRODUCER",
                        message=(
                            f"variable '{name}' is produced by more than one tool: "
                            f"{', '.join(tools)}"
                        ),
                        names=(name, *tools),
                    )
                )
            if kinds[name] in ("range", "choice"):
                code, what = "PRODUCED_DESIGN_VAR", "a design variable"
            elif kinds[name] == "fixed":
                code, what = "PRODUCED_FIXED", "a fixed parameter"
            else:
                continue
            findings.extend(
                Finding(
                    code=code,
                    message=f"tool '{tool}' outputs '{name}', which is {what}",
                    names=(name, tool),
                )
                for tool in tools
            )
        return findings

    def variable(self, name: str) -> Variable:
        """Return the variable called ``name``.

        Args:
            name: Variable name.

        Returns:
            The matching variable.

        Raises:
            KeyError: If no variable has this name.
        """
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(name)

    def tool(self, name: str) -> ToolSpec:
        """Return the tool called ``name``.

        Args:
            name: Tool name.

        Returns:
            The matching tool.

        Raises:
            KeyError: If no tool has this name.
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(name)

    def producers(self) -> dict[str, str]:
        """Map every produced variable name to the tool that produces it.

        Returns:
            Dictionary of variable name to producing tool name.
        """
        return {name: tool.name for tool in self.tools for name in tool.outputs}

    def content_hash(self) -> str:
        """Hash the schema content, sensitive to variable and tool order.

        Returns:
            SHA-256 hex digest of the canonical JSON form.
        """
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def report_from_validation_error(exc: ValidationError) -> ValidationReport:
    """Convert a failed ``StudySchema`` validation into a report.

    Args:
        exc: Error raised by ``StudySchema.model_validate``.

    Returns:
        Report with one finding per structural violation, and a
        ``SCHEMA_INVALID`` finding for every other pydantic error.
    """
    findings: list[Finding] = []
    for error in exc.errors(include_url=False):
        if error["type"] == "study_structure":
            findings.extend(Finding(**item) for item in error["ctx"]["findings"])
            continue
        location = error["loc"]
        findings.append(
            Finding(
                code="SCHEMA_INVALID",
                message=f"{'.'.join(map(str, location))}: {error['msg']}",
                names=tuple(part for part in location if isinstance(part, str)),
            )
        )
    return ValidationReport(errors=tuple(findings))
