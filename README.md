# Arke

> AI-First Operator Description Language & Compiler Toolchain

**Arke** (*/ˈɑːrki/*) is an AI-native domain-specific language for describing and optimizing tensor operators. Named after the Greek messenger goddess who bridges Olympus and the mortal world, Arke connects AI intelligence with hardware compute.

## Vision

Current AI compiler stacks (TVM, MLIR, Triton) were designed for human programmers. Arke is designed **AI-First** — optimized for LLM agents to read, write, and reason about operator optimizations, while remaining human-readable and Python-interoperable.

## Key Features

- 🤖 **AI-First Design** — Explicit semantics, structured representation, enumerable search spaces
- 🔗 **Language + IR** — A DSL with a multi-level IR backend (Semantic Graph → Schedule Tree → Hardware Mapping)
- 🐍 **Python Interop** — Bidirectional conversion with Python DSL
- 💬 **`@rationale` Annotations** — Every optimization decision carries a natural language explanation
- 🎯 **Multi-Target** — NVIDIA Ampere, Huawei Ascend A3 (extensible)

## Quick Example

```arke
// Declare computation (what to compute)
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

// Declare optimization strategy (how to optimize)
schedule fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");
    tile(loop="j", factors=[128, 8])
        @rationale("maximize memory coalescing");
    fuse(ops=["matmul", "relu"], type=epilogue);
}
```

## Python DSL

```python
import arke

@arke.kernel
def fused_matmul_relu(
    A: arke.Tensor[1024, 512, arke.f16],
    B: arke.Tensor[512, 2048, arke.f16]
) -> arke.Tensor[1024, 2048, arke.f16]:
    C = arke.matmul(A, B)
    return arke.relu(C)

# Get IR and optimize
ir = fused_matmul_relu.to_ir()
with arke.schedule(ir, target="nvidia_ampere") as s:
    s.tile("i", [64, 16])
    s.fuse("matmul", "relu")

# Generate code
code = ir.codegen(target="triton")
```

## Architecture

```
Human / AI Agent
    │
    ▼
  Arke Language (syntax, semantics, type system)
    │ parse
    ▼
  Arke IR — High Level (Semantic Graph)
    │ optimize
    ▼
  Arke IR — Low Level (Schedule Tree)
    │ codegen
    ▼
  Hardware Code (CUDA / Ascend)
```

## CLI

```bash
arke parse kernel.ak -o kernel.json     # Parse to IR
arke inspect kernel.json                 # View IR
arke optimize kernel.json --target ampere # AI-guided optimization
arke codegen kernel.json --target triton  # Code generation
arke verify kernel.json --ref kernel.py   # Correctness check
```

## Project Status

🚧 **Early development** — Not yet ready for production use.

See [docs/design.md](docs/design.md) for the full design document.

## License

[Apache License 2.0](LICENSE)
