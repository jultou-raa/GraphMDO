"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from mdo_framework.schema import DesignVariable, FixedParam, StateVar, StudySchema


@dataclass(frozen=True)
class DependencyWalk:
    """Everything a set of target outputs depends on in a study.

    Variables and tools follow the order of the schema (FalkorDB insertion
    order), never the traversal or alphabetical order.

    Attributes:
        design_variables: Required range and choice variables.
        fixed_parameters: Required fixed parameters.
        tools: Names of the required tools.
        unknown_targets: Targets that are not declared variables, in target order.
        unproduced_targets: Declared targets that no tool produces, in target order.
        unproduced_states: Required state inputs that no tool produces.
        couplings: State variables exchanged inside a tool cycle, over the whole
            schema and not only the required tools.
    """

    design_variables: tuple[DesignVariable, ...]
    fixed_parameters: tuple[FixedParam, ...]
    tools: tuple[str, ...]
    unknown_targets: tuple[str, ...]
    unproduced_targets: tuple[str, ...]
    unproduced_states: tuple[str, ...]
    couplings: tuple[str, ...]


def walk_dependencies(schema: StudySchema, targets: Sequence[str]) -> DependencyWalk:
    """Resolve the tools and inputs required to compute the target outputs.

    Args:
        schema: Study whose variables and tools are traversed.
        targets: Names of the variables to compute.

    Returns:
        The required design variables, fixed parameters and tools, the targets
        and inputs that cannot be resolved, and the coupling variables.
    """
    variables = {variable.name: variable for variable in schema.variables}
    producers = schema.producers()
    tools = {tool.name: tool for tool in schema.tools}
    unique_targets = list(dict.fromkeys(targets))

    required_tools: set[str] = set()
    required_inputs: set[str] = set()
    unproduced_states: set[str] = set()
    pending = [producers[name] for name in unique_targets if name in producers]
    while pending:
        tool_name = pending.pop()
        if tool_name in required_tools:
            continue
        required_tools.add(tool_name)
        for input_name in tools[tool_name].inputs:
            if not isinstance(variables[input_name], StateVar):
                required_inputs.add(input_name)
            elif input_name in producers:
                pending.append(producers[input_name])
            else:
                unproduced_states.add(input_name)

    ordered_inputs = [
        variable for variable in schema.variables if variable.name in required_inputs
    ]
    return DependencyWalk(
        design_variables=tuple(
            variable
            for variable in ordered_inputs
            if not isinstance(variable, FixedParam)
        ),
        fixed_parameters=tuple(
            variable for variable in ordered_inputs if isinstance(variable, FixedParam)
        ),
        tools=tuple(tool.name for tool in schema.tools if tool.name in required_tools),
        unknown_targets=tuple(name for name in unique_targets if name not in variables),
        unproduced_targets=tuple(
            name
            for name in unique_targets
            if name in variables and name not in producers
        ),
        unproduced_states=tuple(
            variable.name
            for variable in schema.variables
            if variable.name in unproduced_states
        ),
        couplings=_find_couplings(schema),
    )


def _find_couplings(schema: StudySchema) -> tuple[str, ...]:
    """Find the variables exchanged between tools of the same cycle."""
    producers = schema.producers()
    graph = nx.DiGraph()
    graph.add_nodes_from(tool.name for tool in schema.tools)
    graph.add_edges_from(
        (producers[name], tool.name)
        for tool in schema.tools
        for name in tool.inputs
        if name in producers
    )
    cycle_of = {
        tool_name: index
        for index, component in enumerate(nx.strongly_connected_components(graph))
        if len(component) > 1
        for tool_name in component
    }
    coupled = {
        name
        for tool in schema.tools
        for name in tool.inputs
        if name in producers
        and producers[name] in cycle_of
        and cycle_of[producers[name]] == cycle_of.get(tool.name)
    }
    return tuple(
        variable.name for variable in schema.variables if variable.name in coupled
    )
