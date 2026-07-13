# Arke Harness — MCP Interface Reference

**Protocol:** MCP (Model Context Protocol, version `2024-11-05`)
**Transports:** stdio (local) + Streamable HTTP (remote)
**Implementation:** `arke/agent/mcp_server.py` (zero external deps, stdlib only)

---

## 1. Capabilities advertised

```json
{
  "tools":     { "listChanged": false },
  "resources": { "listChanged": false, "subscribe": false },
  "prompts":   { "listChanged": false }
}
```

---

## 2. MCP methods supported

| Method | Type | Purpose |
|--------|------|---------|
| `initialize` | request | Server capabilities + serverInfo (name, version, op, target_hw) |
| `ping` | request | Liveness probe |
| `tools/list` | request | 8 frozen Façade tools (name + description + inputSchema) |
| `tools/call` | request | Execute one tool, return ToolResult JSON |
| `resources/list` | request | 4 readable optimization-context resources |
| `resources/read` | request | Read a resource by URI |
| `prompts/list` | request | 2 reusable optimization prompt templates |
| `prompts/get` | request | Render a prompt template with arguments |
| `notifications/initialized` | notification | ACK, no response |

Unknown methods → JSON-RPC `-32601 method not found`.

---

## 3. Tools (frozen Façade v1.0)

| Tool | Budget | Purpose |
|------|--------|---------|
| `get_hw_profile` | free | Target HW constraints |
| `analyze_compute` | free | Kernel compute characteristics |
| `list_legal_actions` | free | Compiler-legal move set |
| `apply_decision` | decision | Apply a legal move + @rationale |
| `verify_correctness` | compile | V1 numeric correctness |
| `compile_and_profile` | compile | V2 real-GPU latency + robust_reward |
| `checkpoint` | free | Snapshot current strategy |
| `rollback` | free | Restore a prior checkpoint |

---

## 4. Resources

| URI | Content | Updates |
|-----|---------|---------|
| `arke://context/op` | `{op, shapes, target_hw}` | static per session |
| `arke://hw/profile` | Hardware profile dict | static per session |
| `arke://strategy/current` | Full StrategyIR + decision_log + budget | mutated by apply_decision |
| `arke://actions/legal` | Compiler-legal actions | changes after each decision |

Read via `resources/read` with `{"uri": "arke://..."}`; returns `contents[].text` (JSON).

---

## 5. Prompts

| Name | Arguments | Purpose |
|------|-----------|---------|
| `optimize_kernel` | `target_ratio` (optional, default "1.0") | Guide agent through the bounded optimization loop |
| `explain_strategy` | — | Summarize the current StrategyIR + @rationale trail |

Get via `prompts/get` with `{"name": "...", "arguments": {...}}`.

---

## 6. Transports

### 6a. stdio (local, default)

```bash
arke mcp serve --kernel matmul --shape 512,512,512
# → line-delimited JSON-RPC 2.0 over stdin/stdout
```

### 6b. Streamable HTTP (remote)

```bash
arke mcp serve --kernel matmul --shape 512,512,512 --http --port 8765
# → POST http://127.0.0.1:8765/mcp (JSON-RPC body, JSON-RPC response)
# → GET  http://127.0.0.1:8765/health (liveness: {server, version, transport, op})
```

### 6c. SSE Streaming (progressive)

To get **progress events** during expensive tool calls (compile_and_profile,
verify_correctness), add `Accept: text/event-stream` to the POST:

```bash
curl -N -X POST http://127.0.0.1:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"compile_and_profile","arguments":{}}}'
# Response (SSE):
# event: progress
# data: {"stage":"starting","tool":"compile_and_profile","message":"Starting..."}
#
# event: progress
# data: {"stage":"compiling","tool":"compile_and_profile","message":"Compiling..."}
#
# event: progress
# data: {"stage":"complete","tool":"compile_and_profile","message":"...completed."}
#
# event: result
# data: {"jsonrpc":"2.0","id":1,"result":{...}}
```

Without `Accept: text/event-stream`, the same POST returns a plain JSON response
(no streaming). Cheap tools (`get_hw_profile`, `list_legal_actions`, ...) emit
only the `result` event — no progress frames.

Notifications (`"id"` absent) → HTTP 202 Accepted, empty body.
Bad paths → 404. Parse errors → standard JSON-RPC error envelope.

Use `--host 0.0.0.0` to expose to a network (e.g. containers, remote agents).

---

## 7. Compatible agent systems (stdio or HTTP)

Any MCP-compatible client can connect. Verified / documented:

| Agent System | Transport | Integration |
|---|---|---|
| **Claude Code** | stdio | `claude mcp add arke -- arke mcp serve --kernel <op>` |
| **Claude Desktop** | stdio | `claude_desktop_config.json` → `mcpServers.arke` |
| **Hermes** | stdio | MCP servers config `command: arke, args: [mcp, serve, ...]` |
| **OpenClaw** | stdio | MCP servers list registration |
| **Cline** (VSCode) | stdio | MCP server settings |
| **Continue** (VSCode/JetBrains) | stdio | `config.json` mcpServers |
| **Cursor** | stdio | MCP config |
| **Zed** | stdio | Assistant context servers |
| **OpenAI Agents SDK** | stdio/HTTP | MCP connector (stdio or streamable-http URL) |
| **LangGraph / LangChain** | stdio/HTTP | `langchain-mcp-adapters` |
| **Google ADK / Goose** | stdio | MCP stdio client |

### Quick start for Hermes:
```bash
# In Hermes MCP config:
arke:
  command: arke
  args: ["mcp", "serve", "--kernel", "matmul", "--shape", "512,512,512"]
```

### Quick start for remote (HTTP) — e.g. from another container:
```bash
# Server (GPU machine):
arke mcp serve --kernel matmul --shape 512,512,512 --http --host 0.0.0.0 --port 8765

# Client (any machine):
curl -X POST http://<server>:8765/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

---

## 8. Test coverage

| File | Tests | Covers |
|------|:-----:|--------|
| `tests/test_mcp_server.py` | 8 | Original tools/list, tools/call, notifications, stdio round-trip |
| `tests/test_mcp_primitives_http.py` | 26 | resources, prompts, capabilities, HTTP transport, **SSE streaming** (progress/result events, expensive vs cheap, notification) |
| `tests/test_rl_quality_m3.py` | 20 | M3 quality gates (schema/dedup/reward/tier) |
| `tests/test_harness_usage_e2e.py` | 5 | 3-mode end-to-end (heuristic/MCP/builtin) |

Total MCP + RL quality: **54 tests** + 5 e2e (all pass, zero external deps).

---

*Last updated: 2026-07-13. Implementation: `arke/agent/mcp_server.py`.*
