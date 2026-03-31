# Arke — 设计修正补丁 v2.1.2

> 两项关键修正
> Date: 2026-03-31

---

## 修正 1：LLM API 灵活切换与工程配置

### 问题

当前设计中 LLM 调用是硬编码的。实际需要：
- 灵活切换 LLM provider（Claude / GPT / Qwen / 本地模型）
- API key、endpoint、model 等配置外置
- 不同 LLM 的 tool-use 协议差异需要适配
- 失败重试、rate limit、超时等工程健壮性

### 设计

#### 1.1 LLM Provider 抽象

```python
# arke/agent/llm_provider.py

@dataclass
class LLMConfig:
    """LLM 配置，从 YAML/TOML 或环境变量加载"""
    provider: str = "openai"         # "openai" | "anthropic" | "local"
    model: str = "gpt-4o"
    api_key: str = ""                # 从环境变量 ARKE_LLM_API_KEY 或配置文件
    base_url: str = ""               # 自定义 endpoint（vLLM/Ollama/LMStudio/云雾 等）
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_sec: int = 120
    max_retries: int = 3
    retry_delay_sec: float = 2.0

@dataclass
class LLMResponse:
    """统一响应格式，屏蔽 provider 差异"""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # 统一格式
    usage: dict = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0

class LLMProvider(ABC):
    """所有 LLM 提供商的基类"""
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] = None) -> LLMResponse: ...
    @abstractmethod
    def supports_tool_use(self) -> bool: ...
```

#### 1.2 多 Provider 实现

```
arke/agent/providers/
├── __init__.py        # 注册表 + 工厂
├── base.py            # LLMProvider ABC + LLMConfig + LLMResponse
├── openai_compat.py   # OpenAI-compatible（也支持 vLLM/Ollama/LMStudio/yunwu.ai 等）
├── anthropic.py       # Anthropic Claude（tool_use 格式不同）
└── fallback.py        # 多 provider 自动 fallback 链
```

**关键：OpenAI-compatible provider 覆盖了 80% 的场景**（GPT、Qwen via DashScope、DeepSeek、本地 vLLM/Ollama 都兼容 OpenAI API），只有 Anthropic 需要单独适配。

#### 1.3 配置文件

```yaml
# arke.config.yaml（项目根目录或 ~/.arke/config.yaml）

llm:
  # 默认 provider
  default: "openai"

  providers:
    openai:
      model: "gpt-4o"
      api_key: "${OPENAI_API_KEY}"     # 支持环境变量引用
      base_url: ""                      # 默认 OpenAI 官方
      temperature: 0.1
      max_tokens: 4096
      timeout_sec: 120
      max_retries: 3

    anthropic:
      model: "claude-sonnet-4-6"
      api_key: "${ANTHROPIC_API_KEY}"
      temperature: 0.1
      max_tokens: 4096

    yunwu:
      # 云雾代理，兼容 OpenAI API
      provider_type: "openai_compat"
      model: "claude-opus-4-6"
      api_key: "${YUNWU_API_KEY}"
      base_url: "https://yunwu.ai/v1"

    qwen:
      provider_type: "openai_compat"
      model: "qwen-max"
      api_key: "${DASHSCOPE_API_KEY}"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

    local:
      provider_type: "openai_compat"
      model: "qwen2.5-14b-instruct"
      base_url: "http://localhost:1234/v1"
      api_key: "not-needed"

  # Fallback 链：如果第一个失败，自动尝试下一个
  fallback_chain: ["yunwu", "anthropic", "qwen", "local"]

# 硬件目标
hardware:
  default: "nvidia_ampere"

# 优化预算
optimization:
  max_decisions: 50
  max_compiles: 10
  target_performance_ratio: 0.7
```

#### 1.4 CLI 集成

```bash
# 使用默认配置
arke optimize kernel.json --target ampere

# 指定 LLM provider
arke optimize kernel.json --target ampere --llm anthropic

# 指定具体模型
arke optimize kernel.json --target ampere --llm yunwu --model claude-opus-4-6

# 使用本地模型
arke optimize kernel.json --target ampere --llm local

# 环境变量覆盖
ARKE_LLM_PROVIDER=anthropic ARKE_LLM_MODEL=claude-opus-4-6 arke optimize kernel.json
```

#### 1.5 工程健壮性

```python
class ResilientLLMProvider:
    """带 fallback 和健壮性的 LLM 调用封装"""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.providers = self._init_providers()
        self.fallback_chain = self.config.fallback_chain
        self.token_tracker = TokenTracker()    # 追踪 token 使用

    def chat(self, messages, tools=None) -> LLMResponse:
        """带 fallback 的 chat 调用"""
        errors = []
        for provider_name in self.fallback_chain:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            try:
                resp = provider.chat(messages, tools)
                self.token_tracker.record(provider_name, resp.usage)
                return resp
            except RateLimitError as e:
                errors.append((provider_name, "rate_limit", str(e)))
                continue  # 尝试下一个
            except TimeoutError as e:
                errors.append((provider_name, "timeout", str(e)))
                continue
            except AuthenticationError as e:
                errors.append((provider_name, "auth", str(e)))
                continue  # API key 失效，跳过
            except Exception as e:
                errors.append((provider_name, "unknown", str(e)))
                raise  # 未知错误直接抛出

        raise AllProvidersFailedError(errors)

class TokenTracker:
    """Token 使用追踪（用于评估和成本分析）"""
    def record(self, provider, usage): ...
    def get_total(self) -> dict: ...
    def get_cost_estimate(self) -> float: ...
```

---

## 修正 2：Ascend 后端策略调整

### 关键发现

调研发现了三个重要信息：

1. **AscendNPU IR**（又名 BiSheng IR）：基于 MLIR 构建的昇腾中间表示，提供多级抽象
   - Linalg/Tensor 层 → HFusion 层 → HIVM 层 → LIR → Binary
   - 支持通过 MLIR 方言直接对接

2. **triton-ascend**：华为官方的 Triton 到 Ascend 的适配器
   - **Triton 代码可以直接跑在 Ascend NPU 上**
   - Triton → MLIR → AscendNPU IR → Binary
   - 已有 matmul 等示例

3. **CCE（Cube Core Engine）**：更底层的编程接口，已逐步被 AscendC 和 AscendNPU IR 取代

### 策略调整

```
旧策略（v2.1.1）：
  NVIDIA 后端：Arke → Triton 代码 → GPU
  Ascend 后端：Arke → AscendC 代码 → NPU    ← 需要全新的模板体系

新策略（v2.1.2）：
  NVIDIA 后端：Arke → Triton 代码 → GPU
  Ascend 后端：Arke → Triton 代码 → triton-ascend → NPU   ← 复用 Triton 代码！
  未来可选：  Arke → AscendNPU IR (MLIR) → NPU             ← 更深度优化
```

**核心洞察：有了 triton-ascend，Arke 的 Triton codegen 就是双硬件通用的。不需要为 Ascend 写一套完全独立的 AscendC 模板。**

### 三层 Ascend 后端路径

```
路径 1（Phase 2，最快落地）：Triton → triton-ascend → NPU
  优点：复用 NVIDIA 的 Triton codegen，零额外 codegen 开发
  缺点：受 triton-ascend 支持的 op/feature 限制
  适用：triton-ascend 已支持的算子（matmul, softmax, attention 等基础算子）

路径 2（Phase 3，中度优化）：Arke → AscendNPU IR HFusion 层
  优点：利用 HFusion 的自动 tiling/scheduling，更贴近硬件
  缺点：需要 MLIR 集成开发
  适用：需要 Ascend 特有优化（如乒乓流水线）的场景

路径 3（Phase 4，极致优化）：Arke → AscendNPU IR HIVM 层
  优点：精确控制 NPU 指令，极致性能
  缺点：工程量大，类似直接写 AscendC
  适用：核心高频算子的极致调优
```

### 更新后的后端架构

```
arke/backend/
├── base.py                  # ArkeBackend ABC
├── registry.py              # 后端注册表
├── hal.py                   # 硬件抽象翻译
├── triton/                  # NVIDIA Triton 后端（Phase 1）
│   ├── backend.py
│   ├── compiler.py          # triton.compile → GPU binary
│   ├── profiler.py
│   └── templates/
│       ├── matmul.py.j2
│       └── softmax.py.j2
├── triton_ascend/           # Ascend via triton-ascend（Phase 2）
│   ├── backend.py           # 继承 TritonBackend，改 compile 路径
│   ├── compiler.py          # triton-ascend compile → NPU binary
│   └── profiler.py          # Ascend profiling
├── ascendnpu_ir/            # AscendNPU IR 直接对接（Phase 3）
│   ├── backend.py
│   ├── mlir_gen.py          # 生成 MLIR (HFusion/HIVM)
│   └── compiler.py          # bishengir-compile → NPU binary
└── llm_gen/                 # LLM 生成（实验性）
    └── generator.py
```

### TritonAscendBackend 设计

```python
class TritonAscendBackend(TritonBackend):
    """Ascend 后端——复用 Triton codegen，改 compile 路径"""

    vendor = "huawei"
    name = "triton_ascend"

    def translate_strategy(self, strategy_ir, hw_profile):
        """Ascend 的策略翻译——tile 可以更大（L1=1MB vs shared=48KB）"""
        translated = super().translate_strategy(strategy_ir, hw_profile)

        # Ascend 特有调整
        # 1. triton-ascend 可能需要特殊的 constexpr 配置
        # 2. 某些 Triton op 在 Ascend 上有限制

        return translated

    def generate_code(self, semantic_ir, translated_strategy):
        """生成 Triton 代码——与 NVIDIA 相同"""
        return super().generate_code(semantic_ir, translated_strategy)

    def compile(self, code):
        """使用 triton-ascend 编译"""
        # import triton_ascend
        # 设置 target="ascend"
        # 编译流程：Triton Python → MLIR → AscendNPU IR → Binary

    def profile(self, kernel, inputs, warmup=5, runs=20):
        """Ascend NPU profiling"""
        # 使用 torch_npu 或 CANN profiling 接口

    def get_baseline(self, semantic_ir, inputs):
        """CANN kernel library baseline"""
```

### HAL 策略调整

由于 NVIDIA 和 Ascend 都走 Triton codegen，**LLM 的策略搜索在两种硬件上的差异缩小了**：

```
之前：
  LLM 为 NVIDIA 做 Triton 策略 → 生成 Triton 代码
  LLM 为 Ascend 做 AscendC 策略 → 生成 AscendC 代码    ← 完全不同的 codegen

现在：
  LLM 为 NVIDIA 做策略 → 生成 Triton 代码 → GPU
  LLM 为 Ascend 做策略 → 生成 Triton 代码 → triton-ascend → NPU
                                            ↑ 同一份 Triton 代码！

差异仅在于：
  - tile 大小不同（Ascend L1 更大 → tile 可以更大）
  - 某些 Triton op 在 Ascend 上可能不支持
  - profiling 接口不同
```

**这大幅简化了多硬件支持的工程量。**

### Ascend 后端落地时间调整

```
Phase 1（Week 1-8）：仅 NVIDIA GPU
  不做任何 Ascend 开发
  但架构上预留后端可插拔

Phase 2（GPU 验证通过后）：Ascend via triton-ascend
  需要：Ascend NPU 硬件/远程环境 + triton-ascend 安装
  工作量：编写 TritonAscendBackend（估计 1-2 周）
  验证：同一份 Triton matmul 代码在 Ascend 上跑通

Phase 3（如有需要）：AscendNPU IR 直接对接
  需要：MLIR 开发能力 + AscendNPU IR SDK
  工作量：大（4-8 周）
  仅在 Phase 2 性能不满足时考虑
```

---

## 对 Plan 的影响

### 删除的任务

| 原任务 | 理由 |
|--------|------|
| W1-06c: Ascend A3 HW Profile | Phase 1 不做 Ascend |
| W5-07: AscendC matmul 模板骨架 | 不再用 AscendC，改用 triton-ascend |
| W5-08: AscendC backend 骨架 | 同上 |
| W1-12: 收集 AscendC 开源样例 | 改为 Phase 2 前收集 triton-ascend 样例 |

### 新增的任务

| 新任务 | Week | 说明 |
|--------|:----:|------|
| LLM 配置系统设计 | W1 | LLMConfig + YAML 配置 + 环境变量 |
| LLM Provider 抽象 + OpenAI-compat 实现 | W3 | 在 LLM runner 之前 |
| Anthropic Provider 适配 | W4 | Claude 的 tool_use 格式不同 |
| Fallback 链 + Token 追踪 | W4 | 工程健壮性 |
| arke.config.yaml 配置框架 | W1 | 全局配置加载 |

### 更新后的项目结构（backend 部分）

```
arke/backend/
├── base.py                  # ArkeBackend ABC
├── registry.py              # 后端注册表
├── hal.py                   # 硬件抽象翻译
├── triton/                  # NVIDIA 后端（Phase 1）
│   ├── backend.py
│   ├── compiler.py
│   ├── profiler.py
│   └── templates/
├── triton_ascend/           # Ascend 后端（Phase 2，骨架预留）
│   └── __init__.py          # Phase 1 只有空文件
└── llm_gen/
    └── generator.py

arke/agent/
├── providers/               # LLM Provider（新增）
│   ├── __init__.py          # 注册表 + 工厂
│   ├── base.py              # ABC + Config + Response
│   ├── openai_compat.py     # OpenAI-compatible
│   ├── anthropic.py         # Claude 适配
│   └── fallback.py          # Fallback 链
├── tools_schema.py
├── session.py
├── prompts.py
├── runner.py                # 使用 Provider 抽象，不直接调 API
└── recovery.py
```

---

## 总结

| 修正 | 影响 |
|------|------|
| **LLM Provider 抽象** | 支持灵活切换 Claude/GPT/Qwen/本地，YAML 配置，fallback 链，token 追踪 |
| **Ascend 策略调整** | 从 AscendC → triton-ascend，复用 Triton codegen，大幅减少多硬件工程量 |
| **Phase 1 纯 NVIDIA** | 删除所有 Phase 1 Ascend 任务，Phase 2 再做 |
| **三层 Ascend 路径** | triton-ascend（快）→ AscendNPU IR HFusion（中）→ HIVM（慢），渐进式深入 |

---

*补丁版本：v2.1.2 | 创建日期：2026-03-31*