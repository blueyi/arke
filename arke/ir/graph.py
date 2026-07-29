# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — Minimal IR for S6 Pass Pipeline.

Lightweight IR representation that the pass system operates on.
This expands in S7 into the full multi-layer Lang & IR architecture.

For S6, IR is a graph of nodes where each node is an op invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arke.ir.semantic import SemanticIR
    from arke.ir.strategy import StrategyIR


# ─── dtype vocabulary bridge ───────────────────────────────────────────────
#
# SemanticIR (Layer 4, LLM-facing) uses the compact MLIR-style dtype names
# ("f16", "bf16", "f32", …). The backend IRGraph (the graph the pass system
# and codegen operate on) uses the verbose torch-style names
# ("float16", "bfloat16", "float32", …). K-H1 makes ``from_semantic`` the one
# place this translation happens, instead of every backend entry re-deriving
# an ad-hoc mapping.

_SEM_TO_GRAPH_DTYPE = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "bool": "bool",
}
_GRAPH_TO_SEM_DTYPE = {v: k for k, v in _SEM_TO_GRAPH_DTYPE.items()}


def semantic_dtype_to_graph(dtype: str) -> str:
    """Map a SemanticIR compact dtype to the IRGraph verbose dtype.

    Unknown / already-verbose names pass through unchanged so callers that
    hand-build graphs with torch-style names keep working.
    """
    return _SEM_TO_GRAPH_DTYPE.get(dtype, dtype)


def graph_dtype_to_semantic(dtype: str) -> str:
    """Inverse of :func:`semantic_dtype_to_graph` (for the round-trip)."""
    return _GRAPH_TO_SEM_DTYPE.get(dtype, dtype)


@dataclass
class IRValue:
    """A value (tensor) in the IR graph.

    Identified by a unique name (SSA form: each name defined exactly once).
    """
    name: str
    dtype: str = "float32"  # "float32", "float16", "bfloat16", "int32", "int64", "int8"
    shape: list[int] = field(default_factory=list)
    is_input: bool = False

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IRValue):
            return self.name == other.name
        return NotImplemented


@dataclass
class IRNode:
    """A single operation in the IR graph.

    Represents one op invocation with named inputs and outputs.
    """
    id: str  # unique node identifier
    op: str  # operator name (must exist in OpRegistry)
    inputs: dict[str, str]  # op_input_name -> IRValue.name
    outputs: list[str]  # list of IRValue.name produced
    attrs: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None  # @rationale annotation


@dataclass
class IRGraph:
    """A computation graph in SSA form.

    The graph consists of:
    - values: all tensor values (inputs + intermediate + outputs)
    - nodes: ordered list of operations
    - graph_inputs: names of input values
    - graph_outputs: names of output values
    """
    name: str = "unnamed"
    values: dict[str, IRValue] = field(default_factory=dict)
    nodes: list[IRNode] = field(default_factory=list)
    graph_inputs: list[str] = field(default_factory=list)
    graph_outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_input(self, name: str, dtype: str = "float32", shape: list[int] | None = None) -> IRValue:
        """Add a graph input value."""
        v = IRValue(name=name, dtype=dtype, shape=shape or [], is_input=True)
        self.values[name] = v
        self.graph_inputs.append(name)
        return v

    def add_node(self, node: IRNode) -> IRNode:
        """Add an operation node and register its output values."""
        self.nodes.append(node)
        for out_name in node.outputs:
            if out_name not in self.values:
                self.values[out_name] = IRValue(name=out_name)
        return node

    def set_outputs(self, names: list[str]) -> None:
        """Set graph output values."""
        self.graph_outputs = names

    def get_value(self, name: str) -> IRValue:
        """Get a value by name."""
        if name not in self.values:
            raise KeyError(f"Value {name!r} not found in graph")
        return self.values[name]

    def defined_names(self) -> set[str]:
        """Get all defined value names (inputs + node outputs)."""
        names = set(self.graph_inputs)
        for node in self.nodes:
            names.update(node.outputs)
        return names

    def used_names(self) -> set[str]:
        """Get all used value names (node inputs + graph outputs)."""
        names = set()
        for node in self.nodes:
            names.update(node.inputs.values())
        names.update(self.graph_outputs)
        return names

    # ─── Official SemanticIR bridge (K-H1) ───────────────────────────────
    #
    # ``from_semantic`` is THE canonical SemanticIR → IRGraph construction
    # path. Backend entry points (torch_bridge, agent backends/tools,
    # mlir_gpu preprocessing, …) previously each hand-built a single-node
    # IRGraph with slightly different input-mapping and dtype conventions.
    # Routing them all through this one constructor removes that drift and
    # gives one place to carry dtype / shape / fusion / rationale side info.

    @classmethod
    def from_semantic(
        cls,
        semantic_ir: "SemanticIR",
        strategy_ir: "StrategyIR | None" = None,
        *,
        dim_bindings: dict[str, int] | None = None,
    ) -> IRGraph:
        """Build a backend IRGraph from a SemanticIR (official constructor).

        Args:
            semantic_ir: the Layer-4 operator DAG (LLM-facing).
            strategy_ir: optional StrategyIR — currently used only to stamp
                the kernel's target hardware and decision provenance into
                ``metadata`` so backends can read it without a second arg.
            dim_bindings: optional {symbolic_dim_name: concrete_int} to
                resolve symbolic shapes to concrete ints. Missing symbolic
                dims fall back to the SymbolicDim.default, else are left as
                ``-1`` (unresolved) so downstream shape inference can fill
                them.

        Returns:
            An IRGraph in SSA form with:
              - one IRValue per SemanticIR param (is_input=True)
              - one IRValue per SemanticIR node output (SSA name = node id,
                or ``{node_id}:{port}`` for multi-output ports)
              - one IRNode per SemanticIR Node / MultiOutputNode, with
                inputs mapped {schema_input_name -> producing value name}

        ConditionalNode is not lowered here (it is a shape-regime construct
        resolved earlier by specialization); a graph still carrying one
        raises, so the caller specializes first.
        """
        from arke.ir.semantic import (
            ConditionalNode,
            MultiOutputNode,
            Node,
            NodeRef,
            ParamRef,
            SymbolicDim,
        )

        bindings = dict(dim_bindings or {})
        # Seed bindings from SymbolicDim defaults where not overridden.
        for sd in semantic_ir.symbolic_dims:
            if sd.name not in bindings and sd.default is not None:
                bindings[sd.name] = sd.default

        def _resolve_shape(shape: list) -> list[int]:
            out: list[int] = []
            for d in shape:
                if isinstance(d, int):
                    out.append(d)
                elif isinstance(d, SymbolicDim):
                    out.append(bindings.get(d.name, -1))
                else:
                    out.append(-1)
            return out

        graph = cls(name=semantic_ir.kernel_id or "unnamed")

        # 1. Params → graph inputs.
        for p in semantic_ir.params:
            graph.add_input(
                p.name,
                dtype=semantic_dtype_to_graph(p.dtype),
                shape=_resolve_shape(p.shape),
            )

        # 2. Map SemanticIR value producers to SSA value names.
        #    - a param produces a value named after the param
        #    - a single-output node produces a value named after the node id
        #    - a multi-output node produces one value per port, named
        #      "{node_id}:{port}"
        def _value_name_for_ref(ref) -> str:
            if isinstance(ref, ParamRef):
                return ref.name
            if isinstance(ref, NodeRef):
                return ref.id
            raise ValueError(f"Unsupported InputRef: {ref!r}")

        # 3. Nodes → IRNodes.
        for node in semantic_ir.nodes:
            if isinstance(node, ConditionalNode):
                raise ValueError(
                    "IRGraph.from_semantic cannot lower a ConditionalNode; "
                    "specialize shape regimes before lowering "
                    f"(node id={node.id!r})."
                )

            rationale = None
            if strategy_ir is not None:
                rationale = _rationale_for_node(strategy_ir, node.id)

            in_map = {k: _value_name_for_ref(v) for k, v in node.inputs.items()}

            if isinstance(node, MultiOutputNode):
                out_names = [f"{node.id}:{port}" for port in node.outputs.keys()]
                graph.add_node(IRNode(
                    id=node.id,
                    op=node.op,
                    inputs=in_map,
                    outputs=out_names,
                    attrs=dict(node.attrs),
                    rationale=rationale,
                ))
                for port, desc in node.outputs.items():
                    vname = f"{node.id}:{port}"
                    graph.values[vname] = IRValue(
                        name=vname,
                        dtype=semantic_dtype_to_graph(desc.dtype),
                        shape=_resolve_shape(desc.shape),
                    )
            elif isinstance(node, Node):
                out_name = node.id
                graph.add_node(IRNode(
                    id=node.id,
                    op=node.op,
                    inputs=in_map,
                    outputs=[out_name],
                    attrs=dict(node.attrs),
                    rationale=rationale,
                ))
                graph.values[out_name] = IRValue(
                    name=out_name,
                    dtype=semantic_dtype_to_graph(node.output.dtype),
                    shape=_resolve_shape(node.output.shape),
                )
            else:
                raise ValueError(f"Unsupported SemanticIR node type: {type(node)!r}")

        # 4. Graph outputs from return_node / return_ports.
        if semantic_ir.return_ports:
            # Multi-output return: ports on the return node.
            graph.set_outputs([
                f"{semantic_ir.return_node}:{port}" if f"{semantic_ir.return_node}:{port}" in graph.values
                else port
                for port in semantic_ir.return_ports
            ])
        elif semantic_ir.return_node:
            graph.set_outputs([semantic_ir.return_node])

        # 5. Carry side info into metadata (backends read it optionally).
        graph.metadata = {
            "source": "from_semantic",
            "kernel_id": semantic_ir.kernel_id,
        }
        if semantic_ir.fusion_groups:
            graph.metadata["fusion_groups"] = [
                {"id": fg.id, "nodes": list(fg.nodes), "type": fg.fusion_type}
                for fg in semantic_ir.fusion_groups
            ]
        if strategy_ir is not None:
            graph.metadata["target_hw"] = getattr(strategy_ir, "target_hw", None)

        return graph

    def to_semantic(self) -> "SemanticIR":
        """Reconstruct a SemanticIR from this IRGraph (round-trip inverse).

        Used primarily by the K-H1 round-trip golden test. Recovers params,
        single- and multi-output nodes, per-node output TensorDescs, edges,
        fusion groups, and the return node/ports. Semantics come from the
        OpRegistry (same source ``ast_to_semantic`` uses), so a SemanticIR
        built from the registry round-trips to an equivalent one.
        """
        from arke.ir.semantic import (
            Edge,
            InputRef,
            MultiOutputNode,
            Node,
            NodeRef,
            Param,
            ParamRef,
            Semantics,
            SemanticIR,
            TensorDesc,
        )

        def _sem_semantics(op_name: str) -> Semantics:
            try:
                from arke.ir.ops.registry import REGISTRY
                op = REGISTRY.get(op_name)
                return Semantics(
                    computation=op.computation,
                    index_vars=list(op.index_vars),
                    reduction_axes=list(op.reduction_axes),
                    properties=list(op.properties),
                )
            except (KeyError, ImportError):
                return Semantics(computation=f"{op_name}(...)")

        sem = SemanticIR(kernel_id=self.name)

        # Params from graph inputs.
        input_names = set(self.graph_inputs)
        for name in self.graph_inputs:
            v = self.values[name]
            sem.add_param(Param(
                name=v.name,
                shape=list(v.shape),
                dtype=graph_dtype_to_semantic(v.dtype),
            ))

        # Which value names are produced by which node (for InputRef kind).
        # A value name equal to a param name → ParamRef; else NodeRef whose
        # id is the node id (strip ":port" suffix for multi-output values).
        node_ids = {n.id for n in self.nodes}

        def _mk_ref(value_name: str) -> InputRef:
            if value_name in input_names:
                return ParamRef(name=value_name)
            base = value_name.split(":", 1)[0]
            if base in node_ids:
                return NodeRef(id=base)
            # Unknown producer — treat as param (best effort).
            return ParamRef(name=value_name)

        for node in self.nodes:
            in_refs = {k: _mk_ref(vn) for k, vn in node.inputs.items()}
            semantics = _sem_semantics(node.op)
            if len(node.outputs) > 1 or any(":" in o for o in node.outputs):
                out_descs = {}
                for out_vn in node.outputs:
                    port = out_vn.split(":", 1)[1] if ":" in out_vn else out_vn
                    v = self.values.get(out_vn)
                    out_descs[port] = TensorDesc(
                        shape=list(v.shape) if v else [],
                        dtype=graph_dtype_to_semantic(v.dtype) if v else "f32",
                    )
                sem.add_node(MultiOutputNode(
                    id=node.id,
                    op=node.op,
                    inputs=in_refs,
                    outputs=out_descs,
                    semantics=semantics,
                    attrs=dict(node.attrs),
                ))
            else:
                out_vn = node.outputs[0] if node.outputs else node.id
                v = self.values.get(out_vn)
                sem.add_node(Node(
                    id=node.id,
                    op=node.op,
                    inputs=in_refs,
                    output=TensorDesc(
                        shape=list(v.shape) if v else [],
                        dtype=graph_dtype_to_semantic(v.dtype) if v else "f32",
                    ),
                    semantics=semantics,
                    attrs=dict(node.attrs),
                ))

            # Edges (mirror ast_to_semantic edge convention).
            for inp_ref in in_refs.values():
                if isinstance(inp_ref, ParamRef):
                    from_id = f"param:{inp_ref.name}"
                    tensor = inp_ref.name
                else:
                    from_id = inp_ref.id
                    tensor = inp_ref.id
                sem.add_edge(Edge(
                    from_node=from_id,
                    to_node=node.id,
                    tensor_name=tensor,
                ))

        # Return node/ports from graph outputs.
        if len(self.graph_outputs) == 1:
            out = self.graph_outputs[0]
            sem.return_node = out.split(":", 1)[0]
        elif len(self.graph_outputs) > 1:
            # Multi-output return.
            bases = {o.split(":", 1)[0] for o in self.graph_outputs}
            if len(bases) == 1:
                sem.return_node = bases.pop()
                sem.return_ports = [
                    o.split(":", 1)[1] if ":" in o else o
                    for o in self.graph_outputs
                ]

        return sem


    @classmethod
    def single_node(
        cls,
        op: str,
        shapes: dict[str, list[int]],
        *,
        dtype: str = "float32",
        output_name: str = "output",
        name: str | None = None,
    ) -> IRGraph:
        """Build a single-node IRGraph for ``op`` over named input ``shapes``.

        This is the official replacement for the ad-hoc single-node graphs
        that backend/benchmark/verify entry points used to hand-build (each
        with its own input-mapping convention). It constructs a one-node
        SemanticIR from the op's registry schema and routes it through
        :meth:`from_semantic`, so there is exactly one SemanticIR→IRGraph
        path in the codebase.

        Input mapping: schema input names are paired with the provided
        ``shapes`` keys — positionally when the counts match (tolerating
        callers that key ``shapes`` by their own names), else by identity
        (``shapes`` keys ARE the schema input names). This preserves the
        behavior the scattered call sites relied on.

        Args:
            op: operator name (must exist in the OpRegistry).
            shapes: {input_value_name: shape} for the graph inputs.
            dtype: verbose graph dtype for all inputs + the output value.
            output_name: SSA name for the single output value.
            name: graph name (defaults to ``op``).

        Returns:
            An IRGraph with one input per entry in ``shapes`` and one node.
        """
        from arke.ir.semantic import (
            Node,
            Param,
            ParamRef,
            Semantics,
            SemanticIR,
            TensorDesc,
        )

        sem_dtype = graph_dtype_to_semantic(dtype)
        shape_keys = list(shapes.keys())

        # Resolve the canonical schema input ordering when available.
        try:
            from arke.ir.ops.registry import REGISTRY
            schema = REGISTRY.get(op)
            schema_inputs = list(schema.inputs.keys())
        except (KeyError, ImportError):
            schema_inputs = None

        if schema_inputs and len(schema_inputs) == len(shape_keys):
            # Pair schema inputs with provided value names positionally.
            input_pairs = list(zip(schema_inputs, shape_keys))
        else:
            # Identity: the shapes keys ARE the schema input names.
            input_pairs = [(k, k) for k in shape_keys]

        sem = SemanticIR(kernel_id=name or op)
        for value_name in shape_keys:
            sem.add_param(Param(
                name=value_name,
                shape=list(shapes[value_name]),
                dtype=sem_dtype,
            ))

        node = Node(
            id=output_name,
            op=op,
            inputs={
                schema_name: ParamRef(name=value_name)
                for schema_name, value_name in input_pairs
            },
            output=TensorDesc(shape=[], dtype=sem_dtype),
            semantics=Semantics(computation=f"{op}(...)"),
        )
        sem.add_node(node)
        sem.return_node = output_name

        graph = cls.from_semantic(sem)
        # Rename the single output value to the requested output_name (the
        # SemanticIR node id doubles as the SSA value name; callers expect a
        # stable "output" name and set_outputs already points at it).
        return graph


def _rationale_for_node(strategy_ir: Any, node_id: str) -> str | None:
    """Best-effort extraction of a decision rationale tagged to ``node_id``.

    StrategyIR decisions are not always node-scoped in Phase 1; when a
    decision carries a ``node``/``target`` param matching ``node_id`` and a
    rationale, surface its text. Returns None otherwise.
    """
    decisions = getattr(strategy_ir, "decisions", None)
    if not decisions:
        return None
    for dec in decisions:
        params = getattr(dec, "params", {}) or {}
        target = params.get("node") or params.get("target")
        if target == node_id:
            rat = getattr(dec, "rationale", None)
            text = getattr(rat, "text", None)
            if text:
                return text
    return None
