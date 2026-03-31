# Arke 语法风格对比分析

> 目标：选择最适合 AI-First 定位的语法风格

---

## 候选方案

| 方案 | 风格 | 代表语言 |
|------|------|----------|
| A | Python-like + 大括号 | Rust, Swift, Kotlin |
| B | 纯 Python-like（缩进） | Python, Triton |
| C | C-like | C, CUDA, GLSL |
| D | 声明式 / 配置式 | HCL (Terraform), YAML-like |
| E | 函数式 | Haskell, OCaml, Halide |

---

## 方案 A：Python-like + 大括号

```arke
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

schedule fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");
    fuse(ops=["matmul", "relu"], type=epilogue);
}
```

**优势：**
- ✅ 关键词和类型标注接近 Python，LLM 训练语料丰富
- ✅ 大括号使结构明确，解析器实现简单可靠
- ✅ AI 生成时不会因为缩进错误导致语义变化
- ✅ 嵌套结构清晰（schedule 内嵌套决策块）
- ✅ 适合 JSON/AST 序列化（大括号天然映射树结构）
- ✅ 复制粘贴不丢失结构信息

**劣势：**
- ❌ 比纯 Python 稍显冗余（多了 `{}`、`;`）
- ❌ Python 用户需要适应分号和大括号

**AI 友好度：⭐⭐⭐⭐⭐**
大括号消除了缩进歧义，LLM 生成的代码结构更可靠。

---

## 方案 B：纯 Python-like（缩进敏感）

```arke
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16>:
    let C = matmul(A, B)
    let Y = relu(C)
    return Y

schedule fused_matmul_relu for target("nvidia_ampere"):
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16")
    fuse(ops=["matmul", "relu"], type=epilogue)
```

**优势：**
- ✅ 视觉上最接近 Python，Python 用户零学习成本
- ✅ 代码更简洁，无多余符号
- ✅ LLM 对 Python 缩进风格最熟悉
- ✅ 与 Python DSL 互转换最自然

**劣势：**
- ❌ 缩进敏感 = AI 生成时容易出错（tab vs space、缩进层级错误）
- ❌ 复制粘贴、跨平台传输容易丢失缩进
- ❌ 嵌套深时可读性下降
- ❌ 解析器实现更复杂（需要 INDENT/DEDENT token）
- ❌ JSON/结构化格式转换时缩进信息容易丢失

**AI 友好度：⭐⭐⭐**
LLM 熟悉 Python 缩进，但生成长代码时缩进错误率显著上升。

---

## 方案 C：C-like

```arke
kernel Tensor<[1024,2048], f16> fused_matmul_relu(
    Tensor<[1024,512], f16> A,
    Tensor<[512,2048], f16> B
) {
    Tensor<[1024,2048], f16> C = matmul(A, B);
    Tensor<[1024,2048], f16> Y = relu(C);
    return Y;
}

schedule fused_matmul_relu(target: "nvidia_ampere") {
    tile(i, {64, 16});  // L2 cache line = 64, warp size = 16
    fuse(matmul, relu, EPILOGUE);
}
```

**优势：**
- ✅ CUDA 用户和系统程序员立刻上手
- ✅ 大括号 + 分号，结构极度明确
- ✅ 类型前置，和 C/CUDA 一致

**劣势：**
- ❌ 类型标注冗长（每次都要写完整类型）
- ❌ LLM 训练语料中 C 风格不如 Python 丰富（在 AI 领域）
- ❌ 不支持命名参数（`loop="i"` 这种写法不自然）
- ❌ 与 Python 生态互转换不直观
- ❌ 注释式 rationale（`//` 注释）无法被结构化解析

**AI 友好度：⭐⭐⭐**
结构明确，但语法不是 AI 领域 LLM 最熟悉的。

---

## 方案 D：声明式 / 配置式

```arke
kernel "fused_matmul_relu" {
    input A {
        shape = [1024, 512]
        dtype = f16
        layout = row_major
    }
    input B {
        shape = [512, 2048]
        dtype = f16
        layout = col_major
    }
    output {
        shape = [1024, 2048]
        dtype = f16
    }
    compute {
        C = matmul(A, B)
        Y = relu(C)
    }
}

schedule "fused_matmul_relu" {
    target = "nvidia_ampere"
    tile {
        loop = "i"
        factors = [64, 16]
        rationale = "L2 cache line = 64, warp size = 16"
    }
    fuse {
        ops = ["matmul", "relu"]
        type = "epilogue"
    }
}
```

**优势：**
- ✅ 结构极度清晰，每个字段都有明确的 key-value
- ✅ AI 解析和生成都极其可靠（接近 JSON 的结构化程度）
- ✅ 无歧义，不依赖操作符优先级或缩进
- ✅ 自然映射到 JSON Schema
- ✅ rationale 作为一等字段存在，天然可被结构化提取

**劣势：**
- ❌ 非常冗长，简单的 matmul+relu 需要大量代码
- ❌ 人类写起来繁琐，不适合手写
- ❌ 与 Python 互转换不直观
- ❌ 表达复杂计算时能力不足（循环、条件分支）
- ❌ "看起来像配置文件而非编程语言"，社区吸引力低

**AI 友好度：⭐⭐⭐⭐⭐**
对 AI 最友好，但对人类最不友好。

---

## 方案 E：函数式

```arke
let fused_matmul_relu =
    kernel (A: Tensor<[1024,512], f16>, B: Tensor<[512,2048], f16>)
        -> Tensor<[1024,2048], f16> =
    A |> matmul(B) |> relu

let optimized = fused_matmul_relu
    |> schedule(target="nvidia_ampere")
    |> tile("i", [64, 16], rationale="L2 cache line = 64")
    |> tile("j", [128, 8], rationale="maximize coalescing")
    |> fuse(["matmul", "relu"], epilogue)
```

**优势：**
- ✅ 极度简洁，管道式表达直观
- ✅ 计算和变换都是函数组合，数学上优雅
- ✅ 不可变性天然适合等价性推理
- ✅ 管道操作符 `|>` 对 AI 来说是清晰的操作序列

**劣势：**
- ❌ LLM 对函数式风格不如命令式熟悉
- ❌ Python 用户学习曲线陡峭
- ❌ 复杂的 schedule 组合难以表达
- ❌ 调试困难（管道中间状态不可见）
- ❌ 与 Python 互转换差距大

**AI 友好度：⭐⭐⭐**
简洁但 LLM 训练数据中函数式风格较少。

---

## 综合对比

| 维度 | A Python+括号 | B 纯Python | C C-like | D 声明式 | E 函数式 |
|------|:---:|:---:|:---:|:---:|:---:|
| AI 生成可靠性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| LLM 语料熟悉度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 人类可读性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 人类可写性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Python 互转换 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 解析器可靠性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 表达能力 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 社区吸引力 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| @rationale 集成 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 混合方案考虑

### 方案 A+D（推荐）：Python-like 计算 + 声明式 Schedule

```arke
// 计算层：Python-like + 大括号（方案 A）
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

// 调度层：更偏声明式（方案 D 的清晰度）
schedule fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");

    tile(loop="j", factors=[128, 8])
        @rationale("maximize memory coalescing");

    fuse(ops=["matmul", "relu"], type=epilogue);
}
```

**理由：**
- 计算描述（kernel）需要表达能力 → Python-like
- 优化决策（schedule）需要结构化 → 声明式倾向
- 两者用大括号统一 → 解析可靠
- @rationale 作为一等语法 → AI 学习友好

### 方案 B+D：纯 Python 计算 + 声明式 Schedule

```arke
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16>:
    let C = matmul(A, B)
    let Y = relu(C)
    return Y

schedule fused_matmul_relu for target("nvidia_ampere"):
    tile:
        loop = "i"
        factors = [64, 16]
        rationale = "L2 cache line = 64, warp size = 16"
    fuse:
        ops = ["matmul", "relu"]
        type = epilogue
```

**理由：**
- 最大化 Python 亲和度
- Schedule 部分用缩进式声明，更接近 YAML

---

## 决策建议

**如果 AI 可靠性优先（推荐）：方案 A+D**
- AI 生成最可靠（大括号消除歧义）
- 人类可读性好
- @rationale 集成自然
- Python 互转换可接受

**如果 Python 生态优先：方案 B+D**
- 与 Python 最接近
- 社区吸引力最大
- 但 AI 生成可靠性有风险

---

*待 Leon 决策后更新 Arke-Plan.md*
