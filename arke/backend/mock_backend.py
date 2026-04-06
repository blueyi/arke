# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Backend — MockBackend for testing (S6 Track 3, Task C3.4).

Deterministic backend that executes via SemanticInterpreter (PyTorch eager).
No GPU required — runs on CPU.
"""

from __future__ import annotations

from typing import Any

import torch

from arke.backend.protocol import ArkeBackend, BackendArtifact, CompiledKernel
from arke.ir.graph import IRGraph, IRNode
from arke.ir.ops.interpreter import INTERPRETER


class MockBackend:
    """CPU-only backend using SemanticInterpreter for execution.

    Perfect for testing: no GPU, deterministic, validates correctness.
    """

    name = "mock"

    def lower(self, graph: IRGraph) -> BackendArtifact:
        """Generate a readable pseudo-code representation."""
        lines = ["# MockBackend pseudo-code"]
        for node in graph.nodes:
            inputs_str = ", ".join(f"{k}={v}" for k, v in node.inputs.items())
            outputs_str = ", ".join(node.outputs)
            lines.append(f"{outputs_str} = {node.op}({inputs_str})")
        return BackendArtifact(
            source_code="\n".join(lines),
            backend_name=self.name,
            op_name=graph.name,
            metadata={"nodes": [n.id for n in graph.nodes]},
        )

    def compile(self, artifact: BackendArtifact) -> CompiledKernel:
        """Compile by storing node list for later execution."""
        return CompiledKernel.ok(
            fn=None,
            backend_name=self.name,
            node_ids=artifact.metadata.get("nodes", []),
        )

    def run(
        self,
        kernel: CompiledKernel,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute graph node by node via SemanticInterpreter.

        This actually runs the computation — not a mock in that sense.
        """
        if not kernel.success:
            raise RuntimeError(f"Cannot run failed kernel: {kernel.error}")

        # We need the graph to execute — store it during lower/compile
        graph = kernel.metadata.get("_graph")
        if graph is None:
            raise RuntimeError("MockBackend.run requires _graph in metadata")

        return self._execute_graph(graph, inputs)

    def run_graph(self, graph: IRGraph, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convenience: execute a graph directly without lower/compile.

        Args:
            graph: IR graph to execute
            inputs: Dict of input_name -> tensor

        Returns:
            Dict of output_name -> tensor
        """
        return self._execute_graph(graph, inputs)

    def _execute_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Execute IR graph node-by-node using SemanticInterpreter."""
        # Value store: maps value names to tensors
        values: dict[str, torch.Tensor] = dict(inputs)

        for node in graph.nodes:
            # Gather inputs for this node
            node_inputs = {}
            for op_inp_name, value_name in node.inputs.items():
                if value_name not in values:
                    raise RuntimeError(
                        f"Value {value_name!r} not available at node {node.id!r}"
                    )
                node_inputs[op_inp_name] = values[value_name]

            # Execute via interpreter
            result = INTERPRETER.execute(node.op, node_inputs, node.attrs)

            # Store outputs
            for out_name in node.outputs:
                values[out_name] = result

        # Return graph outputs
        return {name: values[name] for name in graph.graph_outputs}

    def supports_op(self, op_name: str) -> bool:
        from arke.ir.ops.registry import REGISTRY
        return op_name in REGISTRY
