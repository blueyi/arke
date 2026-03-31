# Arke — 详细设计方案 v2.1

> 基于 plan-v2.1.md 的详细设计拆解
> 核心原则：LLM 是第一用户，JSON IR 是工作界面，验证是安全网
> Date: 2026-03-31

---

## 一、LLM Agent Protocol 详细设计

### 1.1 设计哲学

```
LLM 不"写代码"。LLM 通过 tool-use 协议与编译器交互。

LLM 的角色：decision maker（决策者）
编译器的角色：decision validator + executor（验证+执行者）
```

### 1.2 Tool 定义规范

遵循 OpenAI function calling 格式，兼容 Anthropic tool_use。

---

#### Tool 1: `create_kernel`

创建 kernel，定义计算语义。描述"算什么"，不描述"怎么算"。

**参数 Schema：**
```json
{
  "name": "create_kernel",
  "parameters": {
    "type": "object",
    "required": ["name", "params", "return_type", "computations"],
    "properties": {
      "name": {"type": "string"},
      "params": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["name", "shape", "dtype"],
          "properties": {
            "name": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
            "dtype": {"type": "string", "enum": ["f16","f32","f64","bf16","i8","i32"]},
            "layout": {"type": "string", "enum": ["row_major","col_major"], "default": "row_major"}
          }
        }
      },
      "return_type": {
        "type": "object",
        "required": ["shape", "dtype"],
        "properties": {
          "shape": {"type": "array", "items": {"type": "integer"}},
          "dtype": {"type": "string"}
        }
      },
      "computations": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["id", "op", "inputs"],
          "properties": {
            "id": {"type": "string"},
            "op": {"type": "string"},
            "inputs": {"type": "object"},
            "params": {"type": "object"}
          }
        }
      }
    }
  }
}
```

**调用示例：**
```json
{
  "name": "fused_matmul_relu",
  "params": [
    {"name": "A", "shape": [1024, 512], "dtype": "f16"},
    {"name": "B", "shape": [512, 2048], "dtype": "f16", "layout": "col_major"}
  ],
  "return_type": {"shape": [1024, 2048], "dtype": "f16"},
  "computations": [
    {"id": "matmul_0", "op": "matmul", "inputs": {"A": "A", "B": "B"}},
    {"id": "relu_0", "op": "relu", "inputs": {"X": "@matmul_0"}}
  ]
}
```

**返回：**
```json
{
  "success": true,
  "kernel_id": "fused_matmul_relu",
  "semantic_ir": { "..." },
  "auto_analysis": {
    "total_flops": 2147483648,
    "memory_bytes": 6291456,
    "arithmetic_intensity": 341.3,
    "bottleneck": "compute_bound",
    "fusion_opportunities": [
      {"nodes": ["matmul_0", "relu_0"], "type": "epilogue",
       "reason": "relu is elementwise, eliminates 4MB intermediate write"}
    ]
  }
}
```

---

#### Tool 2: `get_hw_profile`

获取目标硬件完整参数。

**返回示例 (nvidia_ampere / RTX 3060 Laptop)：**
```json
{
  "name": "nvidia_ampere",
  "compute_capability": "8.6",
  "compute_units": 30,
  "matrix_unit": {
    "name": "tensor_core",
    "shapes": [[16,8,16]],
    "dtypes": ["f16", "bf16", "tf32"]
  },
  "memory_hierarchy": [
    {"name": "register", "size_per_cu": 65536, "latency_cycles": 1},
    {"name": "shared",   "size_per_cu": 49152, "bandwidth_gbps": 19000, "latency_cycles": 20},
    {"name": "l2_cache", "size_total": 3145728, "bandwidth_gbps": 2000},
    {"name": "global",   "bandwidth_gbps": 336, "latency_cycles": 500}
  ],
  "constraints": {
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,
    "max_registers_per_thread": 255,
    "warp_size": 32
  },
  "peak_tflops_f16": 21.7
}
```

---

#### Tool 3: `analyze_compute`

分析计算特征：算术强度、瓶颈类型、融合机会。

**返回：**
```json
{
  "kernel": "fused_matmul_relu",
  "total_flops": 2147483648,
  "bottleneck": "compute_bound",
  "arithmetic_intensity": 341.3,
  "per_op": [
    {"id": "matmul_0", "flops": 2147483648, "category": "compute_bound",
     "data_reuse": {"A": "broadcast_j (2048x)", "B": "broadcast_i (1024x)"},
     "properties": ["associative", "distributive"]},
    {"id": "relu_0", "flops": 2097152, "category": "memory_bound",
     "properties": ["elementwise", "monotonic"]}
  ],
  "fusion_opportunities": [
    {"nodes": ["matmul_0", "relu_0"], "type": "epilogue",
     "benefit": "eliminate 4MB intermediate global write", "estimated_speedup": 1.15}
  ],
  "suggested_priority": ["fuse", "tile", "parallel", "place"]
}
```

---

#### Tool 4: `list_legal_actions`

列出所有合法优化动作，可按 kind 过滤。

**参数：** `kind?` = "tile" | "fuse" | "reorder" | "parallel" | "place" | "compute"

**返回：**
```json
{
  "legal_actions": [
    {
      "id": "tile_i_64_16",
      "kind": "tile",
      "params": {"loop": "i", "factors": [64, 16]},
      "estimated_impact": {
        "shared_memory_delta": "+8KB",
        "data_reuse_delta": "+2x on A",
        "parallelism": "16 blocks in i"
      }
    }
  ],
  "blocked_actions": [
    {"id": "tile_i_512", "blocked_reason": "shared memory would exceed 48KB"}
  ],
  "search_space_size": 47,
  "hint": "For compute-bound matmul, typical good tiles: i=64-128, j=64-256, k=32-64"
}
```

---

#### Tool 5: `apply_decision`

应用一个优化决策。**必须提供 rationale。**

**参数：**
- `kind`: "tile" | "fuse" | "reorder" | "parallel" | "place" | "compute" | "algorithm"
- `params`: 决策参数
- `rationale`: 自然语言理由

**返回：**
```json
{
  "success": true,
  "step": 1,
  "validation": {
    "pass": true,
    "resource_usage": {
      "shared_memory": 0, "shared_memory_limit": 49152,
      "estimated_threads": 128
    }
  },
  "state_delta": {
    "new_loops": ["i_outer (16)", "i_inner (64)"]
  },
  "decisions_so_far": 1
}
```

---

#### Tool 6: `verify_correctness`

编译并验证数值正确性（vs NumPy）。慢操作 ~100ms-1s。

**返回：**
```json
{
  "pass": true,
  "trials": 3,
  "max_absolute_error": 0.00195,
  "tolerance": {"atol": 0.01, "rtol": 0.01},
  "all_finite": true
}
```

---

#### Tool 7: `compile_and_profile`

编译 + GPU profiling。最慢操作 ~1-5s，节约使用。

**返回：**
```json
{
  "success": true,
  "performance": {
    "latency_us": 142.3,
    "tflops": 15.1,
    "roofline_efficiency": 0.70
  },
  "vs_baseline": {
    "cublas_latency_us": 128.5,
    "ratio": 0.903,
    "label": "90.3% of cuBLAS"
  },
  "resource_actual": {
    "shared_memory_bytes": 16384,
    "registers_per_thread": 48,
    "occupancy": 0.625
  }
}
```

---

#### Tool 8-10: `rollback`, `checkpoint`, `restore`

标准的状态管理 tools。用于 LLM 做探索性决策时的安全网。

---

### 1.3 Session 状态机

```
CREATED → [create_kernel] → ANALYZED → [apply_decision]*
    → OPTIMIZING → [verify/profile] → VERIFYING
    → 满意 → FINALIZED
    → 不满意 → 回到 OPTIMIZING
```

### 1.4 错误恢复协议

系统在 LLM 决策失败时自动注入引导：

| 失败类型 | 自动处理 | 注入给 LLM 的信息 |
|----------|---------|-------------------|
| 约束违反 | 自动 rollback | 哪个约束被违反 + 建议 |
| 数值错误 | rollback 到 last-good | 常见原因列表 |
| 性能退化 | 提示但不 rollback | 前后对比 + 资源利用分析 |
| 预算耗尽 | 强制 finalize | 剩余预算提醒 |

### 1.5 System Prompt

```
You are an expert GPU kernel optimizer working with the Arke compiler.

Your role: Make optimization DECISIONS. The compiler validates and executes them.
You don't write code — you describe strategy.

Workflow:
1. analyze_compute() — understand the computation
2. list_legal_actions() — see available optimizations
3. apply_decision() — apply decisions one at a time (with rationale)
4. verify_correctness() — check numerical accuracy
5. compile_and_profile() — measure actual GPU performance
6. Iterate until satisfied or budget exhausted

Decision priority: fusion > tiling > memory placement > parallelization
Always explain your reasoning via rationale.
Budget: {max_decisions} decisions, {max_compiles} compiles.
```

---

## 二、IR 层详细设计

### 2.1 算子目录

每个算子包含：签名、语义公式、代数性质、融合规则、NumPy 参考。

```python
@dataclass(frozen=True)
class OpDefinition:
    name: str
    inputs: dict[str, str]       # {"A": "Tensor[M,K]", "B": "Tensor[K,N]"}
    output: str                   # "Tensor[M,N]"
    computation: str              # "C[i,j] = sum(A[i,k] * B[k,j], axis=k)"
    index_vars: list[str]
    reduction_axes: list[str]
    properties: list[str]         # ["associative", "elementwise", ...]
    category: str                 # "compute_bound" | "memory_bound" | "elementwise"
    can_fuse_as_epilogue: bool
    numpy_ref: str                # "np.matmul(A, B)"
```

**P0 算子表（10 个）：**

| 算子 | 类别 | 语义 | 可融合 | NumPy |
|------|------|------|:---:|-------|
| matmul | compute | C[i,j]=Σ_k A[i,k]*B[k,j] | prologue | np.matmul |
| batch_matmul | compute | C[b,i,j]=Σ_k A[b,i,k]*B[b,k,j] | prologue | np.matmul |
| relu | element | Y=max(X,0) | epilogue | np.maximum(X,0) |
| gelu | element | Y=X*Φ(X) | epilogue | scipy.special |
| add | element | Y=A+B | epilogue | A+B |
| mul | element | Y=A*B | epilogue | A*B |
| softmax | reduce | Y[i,j]=exp(X[i,j])/Σ exp | no | scipy.special.softmax |
| reduce_sum | reduce | Y[i]=Σ_j X[i,j] | no | np.sum |
| reduce_max | reduce | Y[i]=max_j X[i,j] | no | np.max |
| transpose | move | Y[j,i]=X[i,j] | no | X.T |

### 2.2 IR Builder

从 Python 快速构建 SemanticIR，不需要手写 JSON：

```python
# 使用示例
b = KernelBuilder("fused_matmul_relu")
b.add_param("A", [1024, 512], "f16")
b.add_param("B", [512, 2048], "f16")
m = b.add_op("matmul", {"A": "A", "B": "B"})
r = b.add_op("relu", {"X": m})
b.set_return(r, [1024, 2048], "f16")
ir = b.build()  # → SemanticIR JSON dict
```

### 2.3 合法动作枚举引擎

给定 SemanticIR + 当前 StrategyIR + HW Profile，枚举所有合法下一步：

```python
class LegalActionsEngine:
    def enumerate(self, semantic, strategy, hw) -> list[Action]:
        actions = []
        actions += self._enumerate_tiling(semantic, strategy, hw)
        actions += self._enumerate_fusion(semantic, strategy, hw)
        actions += self._enumerate_parallel(semantic, strategy, hw)
        actions += self._enumerate_placement(semantic, strategy, hw)
        actions += self._enumerate_reorder(semantic, strategy, hw)
        return actions

    def _enumerate_tiling(self, semantic, strategy, hw) -> list[Action]:
        """枚举合法 tiling 选项"""
        # 对每个未被 tile 的循环：
        #   - 候选 factors: power_of_2 ∩ [能整除 loop_bound] ∩ [不超硬件约束]
        #   - 估算影响: shared memory delta, data reuse, parallelism

    def _enumerate_fusion(self, semantic, strategy, hw) -> list[Action]:
        """枚举合法融合选项"""
        # 检查算子间的依赖关系和融合规则
        # epilogue fusion: elementwise op 跟在 compute op 后面
        # vertical fusion: 连续的同维度操作

    # ... 类似地枚举 parallel, placement, reorder
```

---

## 三、验证系统详细设计

### 3.1 V0 静态验证器（每次 apply 自动执行，<1ms）

```python
class StaticValidator:
    def validate(self, semantic, strategy, hw) -> ValidationResult:
        checks = []
        checks.append(self._check_shape_consistency(semantic, strategy))
        checks.append(self._check_hw_constraints(strategy, hw))
        checks.append(self._check_transform_legality(semantic, strategy))
        checks.append(self._check_data_dependency(semantic, strategy))
        return ValidationResult(
            pass_=all(c.pass_ for c in checks),
            checks=checks
        )

    def _check_hw_constraints(self, strategy, hw):
        """检查硬件约束"""
        usage = self._estimate_resource_usage(strategy, hw)
        violations = []
        if usage.shared_memory > hw.constraints.max_shared_memory:
            violations.append(f"shared memory {usage.shared_memory} > {hw.constraints.max_shared_memory}")
        if usage.registers_per_thread > hw.constraints.max_registers_per_thread:
            violations.append(f"registers {usage.registers_per_thread} > {hw.constraints.max_registers_per_thread}")
        if usage.threads_per_block > hw.constraints.max_threads_per_block:
            violations.append(f"threads {usage.threads_per_block} > {hw.constraints.max_threads_per_block}")
        return CheckResult("hw_constraints", pass_=len(violations)==0, violations=violations)
```

### 3.2 V1 数值验证器（~100ms-1s）

```python
class NumericalValidator:
    def validate(self, compiled_kernel, semantic_ir, config) -> NumericalResult:
        """编译后的 kernel vs NumPy 参考"""
        ref_fn = self._build_numpy_reference(semantic_ir)

        results = []
        for trial in range(config.num_trials):
            inputs = self._generate_random_inputs(semantic_ir)
            expected = ref_fn(*inputs)
            actual = compiled_kernel(*[torch.tensor(x, device='cuda', dtype=torch.float16) for x in inputs])
            actual_np = actual.cpu().numpy()

            abs_err = np.max(np.abs(actual_np - expected))
            rel_err = np.max(np.abs(actual_np - expected) / (np.abs(expected) + 1e-8))
            results.append(TrialResult(trial, abs_err, rel_err,
                                       pass_=abs_err < config.atol and rel_err < config.rtol))

        return NumericalResult(
            pass_=all(r.pass_ for r in results),
            trials=results,
            all_finite=np.all(np.isfinite(actual_np))
        )

    def _build_numpy_reference(self, semantic_ir) -> callable:
        """从 SemanticIR 自动生成 NumPy 参考实现"""
        # 遍历 nodes，用每个 op 的 numpy_ref 组合出完整函数
```

### 3.3 V2 性能验证器（~1-5s）

```python
class PerformanceProfiler:
    def profile(self, compiled_kernel, semantic_ir, hw, config) -> ProfileResult:
        # Warmup
        inputs = self._generate_inputs(semantic_ir)
        for _ in range(config.warmup_runs):
            compiled_kernel(*inputs)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(config.benchmark_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled_kernel(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms

        latency_ms = np.median(times)
        flops = self._compute_flops(semantic_ir)
        tflops = flops / (latency_ms * 1e-3) / 1e12

        # vs baseline
        baseline = self._run_baseline(semantic_ir, inputs, config.baseline)

        return ProfileResult(
            latency_us=latency_ms * 1000,
            tflops=tflops,
            roofline_efficiency=tflops / hw.peak_tflops,
            vs_baseline=latency_ms / baseline.latency_ms
        )
```

---

## 四、Codegen 详细设计

### 4.1 路径 A：模板翻译

使用 Jinja2 模板，每种 (算子模式, 策略模式) 有一套模板。

**matmul 模板骨架（Triton）：**

```python
# triton_templates/matmul.py.j2
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    {% if fused_epilogue == "relu" %}
    acc = tl.where(acc > 0, acc, 0.0)
    {% endif %}

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
```

**模板引擎：**

```python
class TritonTemplateEngine:
    def generate(self, semantic_ir, strategy_ir) -> str:
        """从 IR 生成 Triton kernel 代码"""
        pattern = self._match_pattern(semantic_ir)  # "matmul", "matmul_relu", "softmax", ...
        template = self._load_template(pattern)

        # 从 strategy_ir 提取模板参数
        params = self._extract_template_params(strategy_ir)
        # e.g., BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, fused_epilogue="relu"

        return template.render(**params)
```

### 4.2 路径 B：LLM 生成

```python
class TritonLLMGenerator:
    def generate(self, semantic_ir, strategy_ir, hw_profile) -> str:
        """让 LLM 根据 IR 生成 Triton kernel"""
        prompt = self._build_codegen_prompt(semantic_ir, strategy_ir, hw_profile)
        code = self._call_llm(prompt)
        return self._extract_code(code)

    def _build_codegen_prompt(self, semantic, strategy, hw):
        return f"""Generate a Triton GPU kernel based on the following specification.

## Computation (Semantic IR)
{json.dumps(semantic, indent=2)}

## Optimization Strategy
{json.dumps(strategy, indent=2)}

## Hardware Target
{json.dumps(hw, indent=2)}

## Requirements
- Use @triton.jit decorator
- Implement the computation exactly as specified in the semantic IR
- Apply the optimization decisions from the strategy IR
- Output type: float16
- Accumulation type: float32
- Include proper masking for boundary conditions

## Output
Return only the complete Python code with the Triton kernel and a launcher function.
"""
```

### 4.3 编译器 + 执行器

```python
class ArkeCompiler:
    def compile(self, triton_code: str, semantic_ir: dict) -> CompiledKernel:
        """编译 Triton 代码并返回可执行 kernel"""
        # 1. 写入临时文件
        # 2. 动态 import
        # 3. 提取 kernel 函数和 launcher
        # 4. 返回 CompiledKernel wrapper

    def run(self, kernel: CompiledKernel, inputs: list[torch.Tensor]) -> torch.Tensor:
        """执行编译后的 kernel"""

    def profile(self, kernel: CompiledKernel, inputs: list[torch.Tensor],
                warmup=5, runs=20) -> ProfileResult:
        """性能 profiling"""
```

---

## 五、评估框架详细设计

### 5.1 评估任务

```python
@dataclass
class BenchmarkTask:
    id: str
    name: str
    op_spec: dict          # 自然语言 + 结构化描述
    shapes: list[dict]     # 不同 shape 配置
    dtype: str
    target_hw: str
    baseline: str          # "cublas" | "pytorch" | "triton_tuned"
    target_ratio: float    # 目标性能比例
```

**任务列表：**

| ID | 算子 | Shape | 基线 | 目标 |
|----|------|-------|------|:---:|
| T1 | matmul | [1024,512]@[512,2048] f16 | cuBLAS | ≥70% |
| T2 | matmul | [4096,4096]@[4096,4096] f16 | cuBLAS | ≥70% |
| T3 | softmax | [1024,2048] f16 | PyTorch | ≥80% |
| T4 | matmul+relu | [1024,512]@[512,2048] f16 | Triton tuned | ≥70% |
| T5 | matmul (small) | [256,256]@[256,256] f16 | cuBLAS | ≥50% |

### 5.2 三组对比

**Group A: LLM + Arke（本项目）**
- LLM 通过 tool-use 优化
- 输入：SemanticIR + HW Profile
- Budget: 50 decisions, 10 compiles

**Group B: LLM Direct Triton**
- LLM 直接写 Triton kernel
- 输入：自然语言描述 + HW 参数
- 允许 3 轮 fix（如果第一次不正确）

**Group C: LLM Direct CUDA**
- LLM 直接写 CUDA kernel
- 同 Group B

### 5.3 评估指标

| 指标 | 权重 | 说明 |
|------|:---:|------|
| 正确率 | 40% | 通过数值验证的比例 |
| 性能 | 30% | vs vendor baseline |
| 一致性 | 15% | 多次运行的方差 |
| Token 效率 | 15% | 总 token 消耗 |

### 5.4 评估报告格式

```json
{
  "benchmark_id": "arke-eval-v1",
  "date": "2026-04-xx",
  "hardware": "RTX 3060 Laptop, CUDA 13.1",
  "results": [
    {
      "task": "T1",
      "group_a_arke": {
        "correctness": true,
        "performance_vs_baseline": 0.82,
        "tokens_used": 12500,
        "decisions": 23,
        "compiles": 4,
        "time_sec": 45
      },
      "group_b_direct_triton": {
        "correctness": true,
        "performance_vs_baseline": 0.65,
        "tokens_used": 8200,
        "fix_rounds": 1,
        "time_sec": 30
      },
      "group_c_direct_cuda": {
        "correctness": false,
        "fix_rounds": 3,
        "failure_reason": "numerical error in reduction"
      }
    }
  ],
  "summary": {
    "arke_avg_performance": 0.78,
    "direct_triton_avg_performance": 0.61,
    "arke_correctness_rate": 1.0,
    "direct_triton_correctness_rate": 0.8,
    "conclusion": "..."
  }
}
```

---

## 六、开发任务拆解（Task List）

以下是完整的开发任务列表，每个任务标注优先级、依赖关系和预估工时。

### Week 1 任务

| ID | 任务 | Stream | 优先级 | 依赖 | 工时 |
|----|------|:------:|:------:|:----:|:----:|
| W1-01 | 创建 venv + 安装 PyTorch/Triton | 环境 | P0 | - | 2h |
| W1-02 | GPU 环境验证脚本 | 环境 | P0 | W1-01 | 1h |
| W1-03 | 算子目录 P0（10个算子定义） | S2 | P0 | - | 4h |
| W1-04 | Semantic IR JSON Schema + 完善序列化 | S2 | P0 | - | 3h |
| W1-05 | Strategy IR JSON Schema | S2 | P0 | - | 2h |
| W1-06 | HW Profile: nvidia_ampere.json（实测参数） | S2 | P0 | W1-01 | 2h |
| W1-07 | Tool-use Schema 定义（全部 10 个 tool） | S1 | P0 | W1-03 | 4h |
| W1-08 | Session 生命周期 + system prompt 模板 | S1 | P0 | W1-07 | 3h |
| W1-09 | IR Builder（从 Python 快速构建 SemanticIR） | S2 | P1 | W1-03,04 | 3h |
| W1-10 | 集成验证：手动构造 matmul IR → JSON 往返 | ALL | P0 | W1-04,09 | 2h |

### Week 2 任务

| ID | 任务 | Stream | 优先级 | 依赖 | 工时 |
|----|------|:------:|:------:|:----:|:----:|
| W2-01 | V0 静态验证器（shape + 约束 + 变换合法性） | S2 | P0 | W1-04,05,06 | 4h |
| W2-02 | V1 数值验证器（NumPy 参考生成 + 对比） | S2 | P0 | W1-03,W1-01 | 4h |
| W2-03 | ArkeEnv 核心框架 | S1 | P0 | W1-07 | 4h |
| W2-04 | 合法动作枚举引擎（tile/fuse 优先） | S1 | P0 | W1-03,04,06 | 6h |
| W2-05 | ArkeEnv observe/apply/rollback 完整实现 | S1 | P0 | W2-03,04,01 | 4h |
| W2-06 | Strategy IR 重命名 (schedule→strategy) + 关联验证 | S2 | P1 | W1-05 | 2h |
| W2-07 | 单元测试：validator + legal_actions | TEST | P0 | W2-01,04 | 3h |

### Week 3 任务

| ID | 任务 | Stream | 优先级 | 依赖 | 工时 |
|----|------|:------:|:------:|:----:|:----:|
| W3-01 | Triton matmul 模板（Jinja2） | S2 | P0 | W1-01 | 4h |
| W3-02 | Triton matmul+relu 融合模板 | S2 | P0 | W3-01 | 2h |
| W3-03 | 模板引擎（strategy→模板参数映射） | S2 | P0 | W3-01,W1-05 | 4h |
| W3-04 | ArkeCompiler（编译 + 加载 + 执行） | S2 | P0 | W3-01 | 4h |
| W3-05 | 端到端集成：手动 IR → 手动 strategy → Triton → GPU | ALL | P0 | W3-03,04 | 3h |
| W3-06 | V2 性能验证器（profiling + vs cuBLAS） | S2 | P0 | W3-04,W1-01 | 3h |
| W3-07 | ArkeEnv 接入 codegen + verify + profile | S1+S2 | P0 | W3-05,06,W2-05 | 4h |
| W3-08 | Triton softmax 模板 | S2 | P1 | W3-01 | 3h |

### Week 4 任务

| ID | 任务 | Stream | 优先级 | 依赖 | 工时 |
|----|------|:------:|:------:|:----:|:----:|
| W4-01 | LLM Agent Runner（支持 Claude API） | S1 | P0 | W2-05,W3-07 | 4h |
| W4-02 | matmul agent demo：完整 tool-use 循环 | S1 | P0 | W4-01 | 4h |
| W4-03 | 错误恢复模块（约束违反/数值错误/性能退化） | S1 | P0 | W4-01 | 3h |
| W4-04 | 评估任务定义 T1-T5 | S4 | P1 | - | 2h |
| W4-05 | cuBLAS/PyTorch baseline 实现 | S4 | P1 | W1-01 | 2h |
| W4-06 | softmax agent demo | S1 | P1 | W3-08,W4-01 | 3h |
| W4-07 | 多 LLM 后端支持（Qwen/GPT 适配） | S1 | P2 | W4-01 | 3h |

### Week 5 任务

| ID | 任务 | Stream | 优先级 | 依赖 | 工时 |
|----|------|:------:|:------:|:----:|:----:|
| W5-01 | 路径 B：LLM Triton codegen（实验） | S2 | P1 | W3-04,W4-01 | 4h |
| W5-02 | 路径 A vs B 对比实验 | S4 | P1 | W5-01,W3-05 | 3h |
| W5-03 | .ak EBNF 语法定义 (arke.lark) | S3 | P1 | - | 4h |
| W5-04 | Lark Parser 实现 | S3 | P1 | W5-03 | 4h |
| W5-05 | AST → SemanticIR 转换 | S3 | P1 | W5-04,W1-04 | 3h |
| W5-06 | 解析 examples/*.ak → IR → codegen | S3 | P1 | W5-05,W3-03 | 2h |

### Week 6 任务

| ID | 任务 | Stream | 优先级 | 依赖 | 工时 |
|----|------|:------:|:------:|:----:|:----:|
| W6-01 | 端到端 pipeline 串联 | ALL | P0 | W5-05,W4-02 | 4h |
| W6-02 | Group B baseline：LLM 直写 Triton | S4 | P0 | W4-04 | 3h |
| W6-03 | Group C baseline：LLM 直写 CUDA | S4 | P1 | W4-04 | 3h |
| W6-04 | 运行 T1-T3 对比实验 | S4 | P0 | W6-01,02 | 4h |
| W6-05 | fused_matmul_relu 完整端到端 | S2 | P0 | W3-02,W6-01 | 3h |
| W6-06 | 实验数据分析 + 初步结论 | S4 | P1 | W6-04 | 2h |

### Week 7 任务

| ID | 任务 | Stream | 优先级 | 依赖 | 工时 |
|----|------|:------:|:------:|:----:|:----:|
| W7-01 | 轨迹记录系统 | S1 | P1 | W4-02 | 4h |
| W7-02 | 运行 T4-T5 + 多 LLM 对比 | S4 | P1 | W4-07,W6-01 | 4h |
| W7-03 | CLI 完善（parse/inspect/optimize/codegen） | S3 | P1 | W5-04,W6-01 | 4h |
| W7-04 | IR 可视化（inspect --visual） | S3 | P2 | W7-03 | 3h |
| W7-05 | 整体集成测试 + bug 修复 | ALL | P0 | ALL | 4h |

### Week 8 任务

| ID | 任务 | Stream | 优先级 | 依赖 | 工时 |
|----|------|:------:|:------:|:----:|:----:|
| W8-01 | 完整评估报告 | S4 | P0 | W7-02 | 4h |
| W8-02 | 文档完善（agent-protocol/, ir-spec/） | ALL | P1 | ALL | 4h |
| W8-03 | README 更新 + Quick Start 指南 | ALL | P0 | W6-01 | 2h |
| W8-04 | 代码清理 + ruff/mypy 通过 | ALL | P1 | ALL | 3h |
| W8-05 | 测试覆盖率 ≥ 60% | TEST | P1 | ALL | 4h |
| W8-06 | **MVP v0.1.0 Tag** 🎉 | ALL | P0 | ALL | 1h |

---

## 七、关键里程碑

| 里程碑 | 时间 | 验收标准 |
|--------|:----:|----------|
| **M1: IR 可用** | W1 末 | SemanticIR + StrategyIR JSON 往返正确 |
| **M2: 验证可用** | W2 末 | V0 静态验证 + V1 数值验证通过 matmul 测试 |
| **M3: Codegen 可用** | W3 末 | matmul IR → Triton → GPU 执行 → 正确结果 |
| **M4: LLM 循环可用** | W4 末 | LLM 通过 tool-use 完成 matmul 优化端到端 |
| **M5: Parser 可用** | W5 末 | .ak 文件 → IR → codegen → 执行 |
| **M6: 评估完成** | W7 末 | Arke vs 直写 Triton 有定量对比数据 |
| **M7: MVP v0.1.0** | W8 末 | 全部 7 条成功标准达成 |

---

## 八、技术选型确认

| 组件 | 选型 | 理由 |
|------|------|------|
| Parser | Lark (EBNF) | Python 生态最成熟的 parser generator |
| Template | Jinja2 | 已在 dependencies 中（Triton 自带） |
| CLI | Click | 已在 pyproject.toml 中 |
| 序列化 | JSON (stdlib) | LLM 最可靠的结构化格式 |
| GPU 框架 | PyTorch + Triton | NVIDIA 主路径 |
| LLM 调用 | OpenAI-compatible API | 支持 Claude/GPT/Qwen |
| 测试 | pytest | 已配置 |
| Profiling | torch.cuda.Event | 标准 CUDA timing |

---

*详细设计版本：v2.1 | 创建日期：2026-03-31*
*总计约 55 个开发任务，预估 ~200 工时，8 周 MVP*