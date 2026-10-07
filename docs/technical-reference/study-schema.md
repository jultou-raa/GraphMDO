# Study Schema

The `StudySchema` is the typed contract between the graph layer and everything downstream. `GraphManager.get_study_schema()` and the Graph Service `GET /schema` produce it. Topology analysis, the GEMSEO translator, the preflight and the three services consume it. It is defined in `mdo_framework.schema` as frozen Pydantic models that reject unknown fields.

!!! warning "Breaking change"
    The untyped dictionary schema (`get_graph_schema()`, `param_type`, tuple-returning `resolve_dependencies`) is gone, with no compatibility layer. Variable nodes stored in FalkorDB without a `kind` are rejected with a `LEGACY_NODE` finding. Delete and recreate them with the typed API.

## Example

```python
from mdo_framework.schema import (
    ChoiceVar,
    FixedParam,
    ObjectiveSpec,
    RangeVar,
    StateVar,
    StudySchema,
    ToolSpec,
)
from mdo_framework.validation import validate_study

schema = StudySchema(
    variables=[
        RangeVar(name="thickness", lower=0.5, upper=5.0, units="mm"),
        ChoiceVar(name="n_ribs", choices=[2, 3, 4]),
        FixedParam(name="density", value=7850.0, units="kg/m3"),
        StateVar(name="mass", units="kg"),
    ],
    tools=[
        ToolSpec(
            name="Beam",
            inputs=["thickness", "n_ribs", "density"],
            outputs=["mass"],
        )
    ],
)

report = validate_study(schema, objectives=[ObjectiveSpec(name="mass")])
print(report.valid)  # True
```

The same study as JSON (`schema.model_dump(mode="json")`, which is also what `GET /schema` returns):

```json
{
  "schema_version": "1",
  "variables": [
    {
      "kind": "range",
      "name": "thickness",
      "lower": 0.5,
      "upper": 5.0,
      "value_type": "float",
      "initial": null,
      "units": "mm"
    },
    {
      "kind": "choice",
      "name": "n_ribs",
      "choices": [2, 3, 4],
      "initial": null,
      "units": null
    },
    {
      "kind": "fixed",
      "name": "density",
      "value": 7850.0,
      "units": "kg/m3"
    },
    {
      "kind": "state",
      "name": "mass",
      "initial_guess": null,
      "units": "kg"
    }
  ],
  "tools": [
    {
      "name": "Beam",
      "fidelity": "high",
      "inputs": ["thickness", "n_ribs", "density"],
      "outputs": ["mass"]
    }
  ]
}
```

In JSON, `kind` is required on every variable: it selects the model. Optional fields can be omitted from request bodies.

## Variable kinds

Every variable has a `name` (see [Names](#names)) and an optional `units` label. Order is kept: variables, and so design variables, keep the order in which they were created.

| `kind` | Role | Fields | Rules |
| --- | --- | --- | --- |
| `range` | Design variable | `lower`, `upper`, `value_type` (`"float"` default, or `"int"`), `initial` | `lower < upper`. `int` needs integral bounds and `initial`. `initial` lies within the bounds. |
| `choice` | Design variable | `choices`, `initial` | At least 2 choices of one type (`bool`, `int`, `float` or `str`), no duplicates. `initial` is one of the choices. |
| `fixed` | Constant | `value` | `value` is a finite number, a boolean or a string. |
| `state` | Output or coupling | `initial_guess` | `initial_guess` is a finite number, a non-empty list of finite numbers, or `null`. |

Numbers are finite: `NaN` and infinities are rejected, as are booleans or strings where a float is expected.

`initial` is the declared starting value of a design variable. It is checked by the preflight (bounds, and `INITIAL_OUT_OF_SPACE` against linear parameter constraints). The optimizer does not use it yet and starts from the centre of the design space.

### Roles

-   **Design variables** are `range` and `choice` variables. They are the inputs the optimizer varies. A study's design variables are the ones its objectives and constraints depend on, in schema order.
-   **Fixed parameters** are constants. Their value is supplied to the tools; the optimizer never varies them.
-   **State variables** are tool outputs. A state variable exchanged between tools that depend on each other (a cycle) is a coupling. Its `initial_guess` seeds the multidisciplinary analysis; without one, `0.0` is used (warning `COUPLING_DEFAULT_GUESS`).

## Names

Variable, tool, objective and constraint names share one rule (`Name`):

-   they match `^[A-Za-z_][A-Za-z0-9_]*$`,
-   they have at most `MAX_NAME_LENGTH` (50) characters,
-   they are not a Python keyword (`class`, `None`, `True`, ...).

Names become Python keyword arguments of the tool functions and GEMSEO variable names, which is why they are restricted.

## Tools

A tool is a node with a `name` and a `fidelity` (default `"high"`). `ToolNode` is that stored node: it is what `GraphManager.add_tool()` and the Graph Service `POST /tools` take. `ToolSpec` extends it with `inputs` and `outputs`, the variable names derived from the graph edges, and is what the schema contains. Inputs are unique, outputs are unique, and no name is both an input and an output of the same tool.

Tool functions are called with keyword arguments named after the inputs and return a dictionary of outputs (or a single value for a single output).

## Structural invariants

These hold for every `StudySchema`. Building one that breaks them raises a Pydantic `ValidationError`, and `report_from_validation_error()` turns it into a report with the codes below.

| Code | Meaning |
| --- | --- |
| `DUPLICATE_NAME` | Two variables, or two tools, share a name. |
| `UNDECLARED_REF` | A tool input or output is not a declared variable. |
| `DUPLICATE_PRODUCER` | Two tools output the same variable. |
| `PRODUCED_DESIGN_VAR` | A tool outputs a `range` or `choice` variable. |
| `PRODUCED_FIXED` | A tool outputs a `fixed` variable. |

`GraphManager` refuses the writes that would create these (`409`), and `get_study_schema()` checks again when it reads the graph back.

## Validating a study

`validate_study()` answers "can this study run?" without calling a tool. It collects every finding instead of stopping at the first:

```python
from mdo_framework.schema import ObjectiveSpec
from mdo_framework.validation import validate_study


def beam_func(thickness, n_ribs, density):
    return 0.001 * thickness * n_ribs * density


report = validate_study(
    schema,
    objectives=[ObjectiveSpec(name="mass", minimize=True)],
    parameter_constraints=["thickness <= 4"],
    registry={"Beam": beam_func},  # optional: check tools against functions
)
print(report.valid)  # True
```

-   `ObjectiveSpec(name, minimize=True, threshold=None)` names an output to optimize. `constraints=[ConstraintSpec(name, bound, op="<=")]` names outputs to bound (`op` is `"<="` or `">="`). Thresholds and bounds are finite.
-   `parameter_constraints` are linear inequalities over range design variables, written `<lhs> <= <rhs>` or `<lhs> >= <rhs>`. Each side is a sum of numbers, names and `<number>*<name>` terms joined by `+` or `-`, for example `x + 2*y <= 8`.
-   `registry` maps tool names to callables. Without it the registry checks are skipped; the Optimization Service has no registry, the Execution Service does.

The result is a `ValidationReport` with `errors`, `warnings` and a computed `valid` (true when there are no errors). Each `Finding` has a stable `code`, a human-readable `message` and the `names` involved:

```json
{
  "errors": [
    {
      "code": "UNKNOWN_OUTPUT",
      "message": "'stiffness' is not a declared variable",
      "names": ["stiffness"]
    }
  ],
  "warnings": [],
  "valid": false
}
```

`TopologicalAnalyzer.resolve_dependencies()` and `GraphProblemBuilder.build_problem()` raise `StudyValidationError`, which carries the same report in `.report`.

### Error codes

| Code | Raised when | Fix |
| --- | --- | --- |
| `UNKNOWN_OUTPUT` | An objective or constraint names an undeclared variable. | Declare it, or fix the name. |
| `NOT_PRODUCED` | The target is declared but no tool outputs it. | Connect a tool output to it. A design variable cannot be a target. |
| `UNPRODUCED_STATE` | A required tool takes a `state` variable that no tool produces. | Produce it, or make it a `range`, `choice` or `fixed` variable. |
| `NO_DESIGN_VARIABLES` | The targets do not depend on any design variable. | Connect at least one `range` or `choice` variable to the tools that compute them. |
| `UNREGISTERED_TOOL` | A tool has no function in the registry. | Register a function under the tool's name. |
| `SIGNATURE_MISMATCH` | A registered function cannot be called with the tool's inputs as keyword arguments. | Rename the arguments or the graph inputs so they match. |
| `PARAMETER_CONSTRAINT_INVALID` | A parameter constraint does not parse, names an unknown, non-design or `choice` variable, is not finite, or is rejected by Ax. | Use only range design variables and finite coefficients. |
| `INITIAL_OUT_OF_SPACE` | The `initial` values violate a linear parameter constraint. | Move `initial` inside the constraint, or relax it. |
| `DESIGN_SPACE_INVALID` | GEMSEO rejects a design variable. | Fix the variable's bounds or choices. |
| `SCHEMA_INVALID` | A schema or request body does not parse as a `StudySchema` (wrong type, missing `kind`, unknown field, breaks a field rule). The message gives the path. | Fix the field named in the message. |
| `LEGACY_NODE` | A stored variable node has no `kind`. | Delete and recreate it with the typed API. |

The five structural codes above are errors too.

### Warning codes

| Code | Meaning |
| --- | --- |
| `COUPLING_DEFAULT_GUESS` | A coupling `state` variable has no `initial_guess`; `0.0` is used. |
| `UNUSED_VARIABLE` | No tool uses the variable. |
| `UNUSED_TOOL` | The tool is not needed by the objectives and constraints. Only reported when they resolve. |
| `PARTIAL_INITIAL` | Some design variables have an `initial` and others do not. |
| `SIGNATURE_UNCHECKED` | A registered function accepts `**kwargs` or cannot be inspected. |
| `DEFAULTED_ARG_UNWIRED` | A function argument has a default and is not a graph input; the default is used. |

`SIGNATURE_UNCHECKED` and `DEFAULTED_ARG_UNWIRED` need a `registry`.

## JSON Schema

The JSON Schema of the contract is published as [`study-schema.json`](study-schema.json), for clients in other languages and for editor validation. A test (`tests/test_docs_schema.py`) fails when it drifts from the models. Regenerate it from the repository root with:

```bash
uv run python -c "import json, pathlib; from mdo_framework.schema import StudySchema; pathlib.Path('docs/technical-reference/study-schema.json').write_text(json.dumps(StudySchema.model_json_schema(), indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')"
```

The Python reference is in [Schema](../api/schema.md) and [Validation](../api/validation.md).
