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
      "deterministic": true,
      "thread_safe": false,
      "arg_map": {},
      "inputs": ["thickness", "n_ribs", "density"],
      "outputs": ["mass"]
    }
  ]
}
```

In JSON, `kind` is required on every variable: it selects the model (see [Variable kinds](#variable-kinds)). Optional fields can be omitted from request bodies.

## Variable kinds

There are four kinds, one model each. All models are immutable and reject unknown fields. Every variable has a `kind` that selects the model, a `name` (see [Names](#names)) and an optional free-form `units` label. Order is kept: variables, and so design variables, keep the order in which they were created.

Types used in the tables:

-   `Name` is the [name rule](#names).
-   `float` (finite) is a finite number: `NaN`, infinities, booleans and strings are rejected, integers are accepted and widened.
-   `Scalar` is exactly one of `bool`, `int`, finite `float` or `str`. There is no coercion: `1` stays an `int`, `1.0` is a `float`, and `"1"` is not converted.

When you build the Python models directly, `kind` takes the default shown and need not be passed. In JSON, whether a request body or a `StudySchema` payload, `kind` is required: a variable without it is rejected.

### `RangeVar`

Design variable bounded by `[lower, upper]`.

| Field | Type | Default | Validation rules |
| --- | --- | --- | --- |
| `kind` | `"range"` | `"range"` | Discriminator. |
| `name` | `Name` | required | |
| `lower` | `float` (finite) | required | Inclusive lower bound, strictly below `upper`. |
| `upper` | `float` (finite) | required | Inclusive upper bound, strictly above `lower`. |
| `value_type` | `"float"` or `"int"` | `"float"` | With `"int"`, `lower`, `upper` and `initial` (when given) must be integral. |
| `initial` | `float` (finite) or `null` | `null` | Optional starting point. When given, it lies within `[lower, upper]`. |
| `units` | `str` or `null` | `null` | None. |

### `ChoiceVar`

Categorical design variable.

| Field | Type | Default | Validation rules |
| --- | --- | --- | --- |
| `kind` | `"choice"` | `"choice"` | Discriminator. |
| `name` | `Name` | required | |
| `choices` | list of `Scalar` | required | At least 2 values, all of the same type (`1` and `1.0` are different types), no duplicates. |
| `initial` | `Scalar` or `null` | `null` | Optional starting value. When given, it equals one of `choices` and has the same type. |
| `units` | `str` or `null` | `null` | None. |

The type name of the choices (`bool`, `int`, `float` or `str`) is exposed as the read-only `value_type` property. It is not a field and is not accepted in input.

### `FixedParam`

Constant passed unchanged to the tools that declare it.

| Field | Type | Default | Validation rules |
| --- | --- | --- | --- |
| `kind` | `"fixed"` | `"fixed"` | Discriminator. |
| `name` | `Name` | required | |
| `value` | `Scalar` | required | A `bool`, an `int`, a finite `float` or a `str`. |
| `units` | `str` or `null` | `null` | None. |

Tools receive the declared value and type: non-numeric values (`str`, `bool`) and `int` values are encoded for GEMSEO and decoded before the tool call.

### `StateVar`

Output of a tool, or a coupling variable.

| Field | Type | Default | Validation rules |
| --- | --- | --- | --- |
| `kind` | `"state"` | `"state"` | Discriminator. |
| `name` | `Name` | required | |
| `initial_guess` | `float` (finite), non-empty list of `float` (finite), or `null` | `null` | A list must contain at least one value. Seeds the multidisciplinary analysis of a coupling. |
| `units` | `str` or `null` | `null` | None. |

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

A tool is a node with a `name`, a `fidelity` and three execution options, `deterministic`, `thread_safe` and `arg_map`. `ToolNode` is that stored node: it is what `GraphManager.add_tool()` and the Graph Service `POST /tools` take. `ToolSpec` extends it with `inputs` and `outputs`, the variable names derived from the graph edges, and is what the schema contains. Both models are immutable and reject unknown fields.

| Model | Field | Type | Default | Validation rules |
| --- | --- | --- | --- | --- |
| `ToolNode`, `ToolSpec` | `name` | `Name` | required | |
| `ToolNode`, `ToolSpec` | `fidelity` | `Name` | `"high"` | Fidelity level label. It follows the same rule as a name: pattern, length and not a Python keyword. |
| `ToolNode`, `ToolSpec` | `deterministic` | `bool` | `true` | Strict boolean. `false` marks a tool whose outputs are not a function of its inputs alone; its evaluations are never cached. |
| `ToolNode`, `ToolSpec` | `thread_safe` | `bool` | `false` | Strict boolean. `true` declares that the tool may run concurrently in several threads. |
| `ToolNode`, `ToolSpec` | `arg_map` | dict of `Name` to `Name` | `{}` | Graph input name to Python argument name. Values are unique: two graph inputs cannot be mapped to the same argument. That a key is an input of the tool, and that no unmapped input already bears the argument name, are checked against the graph, see `ARG_MAP_UNKNOWN_INPUT` and `ARG_MAP_COLLISION`. |
| `ToolSpec` only | `inputs` | list of `Name` | `[]` | No duplicates. |
| `ToolSpec` only | `outputs` | list of `Name` | `[]` | No duplicates. No name is both an input and an output of the same tool. |

`inputs` and `outputs` must also name declared variables; that check belongs to the [structural invariants](#structural-invariants) of the whole schema.

### Tool function contract

Tool functions are called with keyword arguments named after the inputs. `arg_map` renames an input on the way in: with `arg_map={"x": "a"}` the graph input `x` is passed as the argument `a`, so one function `f(a)` can serve a tool fed by `x` and another fed by `y`. Inputs absent from `arg_map` keep their name. Two inputs cannot reach the same argument, whether through `arg_map` alone or because one is renamed to the name of another that keeps it: with inputs `x` and `y`, `arg_map={"x": "y"}` is an `ARG_MAP_COLLISION`. Swapping two names, `{"x": "y", "y": "x"}`, is fine. The Graph Service and the graph store `arg_map` as a JSON string, because a graph node property cannot hold a map; the API shows it as an object.

A function returns a dictionary keyed by output name, whose keys are exactly the tool's outputs. A tool with a single output may also return the bare value, which can be a list or an array for a vector. A tuple is always an error, whatever the number of outputs, and a list is never mapped to several outputs by position. Every value must be numeric and finite.

A tool that breaks the contract raises a typed error, a subclass of `EvaluationError` (itself a `ValueError`) that carries the tool name and a stable `code`:

| Error | Code | Raised when |
| --- | --- | --- |
| `ToolExecutionError` | `TOOL_FAILED` | The function raises an exception. The message names the tool and the original exception, which is chained as the cause. `KeyboardInterrupt`, `SystemExit` and `MemoryError` are not wrapped. |
| `ToolOutputError` | `OUTPUT_INVALID` | The function returns `None`, a tuple, a dictionary with missing or unexpected keys, a non-dictionary for several outputs, a value that is not numeric, or a NaN or infinite value. |

Jacobians are approximated by finite differences. Evaluations of a tool are cached unless it is declared `deterministic=false`.

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
| `SIGNATURE_MISMATCH` | A registered function cannot be called with the tool's inputs as keyword arguments, after the `arg_map` renames. | Rename the arguments or the graph inputs, or add an `arg_map` entry, so they match. |
| `ARG_MAP_UNKNOWN_INPUT` | An `arg_map` key is not one of the tool's inputs. Part of the registry check, but reported even when the tool has no registered function. | Fix the key, or connect the variable as an input of the tool. |
| `ARG_MAP_COLLISION` | Two or more inputs of a tool are passed as the same argument (the `arg_map` value, or the input name itself when it is not mapped). Part of the registry check, reported even when the tool has no registered function; the signature of a colliding tool is not checked. `names` holds the tool, the argument, then the colliding inputs. | Map each input to its own argument. |
| `TOOL_NOT_THREAD_SAFE` | `GraphProblemBuilder.build_problem()` is given `MDASettings(n_processes > 1)` and a tool that produces a coupling variable does not declare `thread_safe`. Raised at build time only, never by `validate_study()`. | Declare the tool `thread_safe=True` if it can run concurrently, or keep the default sequential MDA. |
| `PARAMETER_CONSTRAINT_INVALID` | A parameter constraint does not parse, names an unknown, non-design or `choice` variable, is not finite, or is rejected by Ax. | Use only range design variables and finite coefficients. |
| `INITIAL_OUT_OF_SPACE` | The `initial` values violate a linear parameter constraint. | Move `initial` inside the constraint, or relax it. |
| `DESIGN_SPACE_INVALID` | GEMSEO rejects a design variable. | Fix the variable's bounds or choices. |
| `SCHEMA_INVALID` | A `StudySchema` payload fails model validation for a reason other than the structural invariants: wrong type, missing or unknown `kind`, unknown field, a broken field rule. `report_from_validation_error()` emits one finding per Pydantic error, and the message starts with the field path. Typically the body that `/validate` and `/optimize` fetch from the Graph Service. | Fix the field named in the message. |
| `LEGACY_NODE` | A stored variable node has no `kind`. | Delete and recreate it with the typed API. |

The five structural codes above are errors too. Like `SCHEMA_INVALID`, they come from `report_from_validation_error()`, which `/validate` and `/optimize` apply to a fetched `StudySchema` payload that fails model validation.

Two things that look similar are not findings of this table:

-   **HTTP request bodies.** A request body that fails validation (a malformed variable or tool on the Graph Service, a malformed `/validate` or `/optimize` body) is answered with the generic `422` `detail` list of `loc`, `msg` and `type` entries, not with these codes. See [Microservices](microservices.md).
-   **Execution Service `SCHEMA_INVALID`.** Its `422` body `{"detail": {"code": "SCHEMA_INVALID", "report": {...}}}` means that the loaded study failed the tool registry check. The `code` labels the response; the findings inside `report` are `UNREGISTERED_TOOL`, `SIGNATURE_MISMATCH`, `ARG_MAP_UNKNOWN_INPUT` or `ARG_MAP_COLLISION`, not a `SCHEMA_INVALID` finding. A body from the Graph Service that does not parse as a `StudySchema` is a `502` there.

### Warning codes

| Code | Meaning |
| --- | --- |
| `COUPLING_DEFAULT_GUESS` | A coupling `state` variable has no `initial_guess`; `0.0` is used. |
| `UNUSED_VARIABLE` | No tool uses the variable. |
| `UNUSED_TOOL` | The tool is not needed by the objectives and constraints. Only reported when they resolve. |
| `PARTIAL_INITIAL` | Some design variables have an `initial` and others do not. |
| `SIGNATURE_UNCHECKED` | A registered function accepts `**kwargs` or cannot be inspected. |
| `DEFAULTED_ARG_UNWIRED` | A function argument has a default and is not fed by a graph input (after the `arg_map` renames); the default is used. |

`SIGNATURE_UNCHECKED` and `DEFAULTED_ARG_UNWIRED` need a `registry`.

## JSON Schema

The JSON Schema of the contract is published as [`study-schema.json`](study-schema.json), for clients in other languages and for editor validation. A test (`tests/test_docs_schema.py`) fails when it drifts from the models. Regenerate it from the repository root with:

```bash
uv run python -c "import json, pathlib; from mdo_framework.schema import StudySchema; pathlib.Path('docs/technical-reference/study-schema.json').write_text(json.dumps(StudySchema.model_json_schema(), indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')"
```

The Python reference is in [Schema](../api/schema.md) and [Validation](../api/validation.md).
