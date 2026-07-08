# Gemma4-26B-A4B Draft-Model 加速项目 — 总规划

> 目标:在现有 MTP 基线(online **2113 tok/s** / offline **2023 tok/s**, accept length **5.03**)之上,
> 用 **MAI Profile 数据**训练更好的 draft model,把吞吐再提升 **10%–20%**(目标 online ~2325–2536 tok/s)。
> 三条技术路线并行:**EAGLE-3 / MTP / DSpark**。
>
> 纪律:本地是开发机(无 GPU)。所有代码 **git commit(本地)→ 用户切好 SSH 后 push → 服务器/容器内 pull 出真实结果**。
> 本文档不编造任何数字;accept rate / tok/s 一律以服务器实测为准。镜像/环境由用户在容器内处理,我方只出脚本。

---

## 0. 仓库地图(四个 repo,职责分离)

| Repo | 分支 | 职责 | 状态 |
|---|---|---|---|
| `zouxinyi0625-hue/vllm-msn` | `feat/gemma4-12b-fp8-bench` | 推理 / benchmark / 部署 / **本总规划文档** | 已有,勿放训练代码 |
| `zouxinyi0625-hue/speculators` (fork) | `dev/maiprofile`(待建) | **方案1 EAGLE-3 训练** + 参考 MTP head | fork 好,在 main |
| `zouxinyi0625-hue/gemma4-mtp-trainer` | `main` / `dev` | **方案2 MTP 训练(重点)**,自研 | 空仓库,从零搭 |
| `zouxinyi0625-hue/DeepSpec` | `dev/maiprofile_moe`(从 `dev/maiprofile` 切) | **方案3 DSpark MoE 扩展** | dense 12B 已跑通 |

---

## 1. 账号隔离(⛔ 当前 BLOCKER)

- **根因**:Copilot 鉴权跟 `gh` active 账号绑定。切到 `zouxinyi0625-hue`(无 Copilot 席位)→ 聊天 403。
- **方案**:`gh` 永远保持 **microsoft 账号 active**(保 Copilot);**push 走 SSH key**(SSH 只认 key 绑定的 GitHub 账号,与 gh 登录无关)。两通道物理隔离。
- **动作**:
  1. [用户] 把本地公钥加到 `zouxinyi0625-hue` GitHub(Settings→SSH keys):
     `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMTBDBhyk/g7ZxHIVnK4ikvK5uFSFy2caTo/KgegSatF xinyizou`
  2. [我] 把四个 repo 的 remote 改成 SSH(`git@github.com:zouxinyi0625-hue/...`)。
  3. 验证:`ssh -T git@github.com` → `Hi zouxinyi0625-hue!`
- 在此之前:**所有改动留本地,不 push**。

---

## 2. 当前真实基线(vllm-msn RESULTS.md)

线上 service = `26b_e011_mtp`, `google/gemma-4-26B-A4B-it` + 官方 assistant(MTP), `spec_tokens=5`。

| 场景 | Output tok/s | accept length | accept rate |
|---|---:|---:|---:|
| offline `26b_e011_mtp` | **2023.06** | — | — |
| offline `26b_e011_no_mtp` | 1474.46 | — | — |
| **online `26b_e011_mtp`** | **2113.78** | **5.03** | **80.58%** |
| online 12B MTP | 1133.69 | — | — |

per-position acceptance:`pos0 93.61 / pos1 87.08 / pos2 80.45 / pos3 74.23 / pos4 67.54`
**靶子:online 2113 / offline 2023 / accept_len 5.03。达标线 = +10~20% → online 2325–2536。**

---

## 3. 三方式对比矩阵

| 维度 | EAGLE-3 | MTP (Google assistant) | DSpark |
|---|---|---|---|
| 支持 finetune | ✅ speculators 从零训 | ⚠️ 自研(Gemma4 结构≠Qwen3 head) | ✅ DeepSpec 已支持 |
| 官方 release (26B-A4B) | ✅ RedHatAI eagle3 | ✅(=当前基线) | ❌ 仅 12B dense |
| 原生支持 MoE | ✅ | ✅ | ❌ 代码写死 dense |
| vllm serve 可用性 | ✅ 成熟 | ✅ 线上在用 | ⚠️ **PR #47216 未 merge,且仅 dense** |
| 训练 loss | soft CE 蒸馏 + TTT | 自研(参考 DeepSpec 蒸馏) | CE .1 + L1 .9 + confidence |
| 本项目重点 | 中 | **高(重点)** | 高(攻 MoE) |

---

## 4. 方案 1 — EAGLE-3(repo: speculators fork @ dev/maiprofile)

### Step A:先测官方 release model 加速比(仅 26B-A4B)
- 部署 `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`,复用 vllm-msn bench scaffold,同 `sc1_delta_v2.jsonl`、同并发。
- 产出:官方 EAGLE-3 draft 的 tok/s + accept_len,直接对比 2113/5.03。

### Step B:用 maiprofile 数据训练自己的 EAGLE-3(8×A100)
- 建分支 `dev/maiprofile`,接 maiprofile 数据(格式:jsonl → `conversations` 字段,与 DSpark 数据同源)。
- 用 fork 里的 `scripts/prepare_data.py` + `train.py`(在线/离线)。
- 数据路径/设置与 DSpark 一致:`$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/...`。
- 产出:maiprofile 分布上 accept_len 超过官方通用 draft。

### 讨论范围:**仅 26B-A4B**。

---

## 5. 方案 2 — MTP 训练(repo: gemma4-mtp-trainer,⭐ 实验重点)

### 目标:保持 Google 官方 MTP 结构,maiprofile finetune 提 accept rate → 提吞吐。

### 关键理解(transformers v5.10.2 源码)
- 官方 draft = `Gemma4AssistantForCausalLM`(≠ speculators Qwen3 MTP head):
  4 层 dense,hidden 1024,backbone 2816。forward 吃 target 的 `inputs_embeds`+`shared_kv_states`
  → pre_projection(2×2816→1024) → 4 层 decoder → post_projection → lm_head。
- ⚠️ forward 强依赖 `shared_kv_states`+`inputs_embeds`(None 即 raise)→ 训练前必须先搭从 target 抽这些量的通路。

### 参考
- speculators MTP(`src/speculators/models/mtp/core.py`):loss=per-step hard CE + `beta^k` 步权重 + 冻结 embed/lm_head。**结构不符,仅借鉴 loss/步权重/冻结策略。**
- DeepSpec `eagle3/loss.py` soft-CE 蒸馏 —— 提 accept rate 的关键。

### 开发计划(8×A100)
1. `debug_gemma_assistant.py`:加载官方 assistant,打印各层 shape,验证接口可前向。
2. 数据通路:maiprofile prompt → target 生成 → 抽 target hidden/KV → 喂 assistant。
3. 训练循环:保持结构,soft-label 蒸馏 + 多 token 步权重,只训 draft 参数。
4. 服务器训练 → 导出(格式须与官方 assistant 兼容,能被现有 `26b_e011_mtp` vLLM 直接加载)→ 用方案1同套 bench 测。

---

## 6. 方案 3 — DSpark MoE(repo: DeepSpec @ dev/maiprofile_moe)

### 现状
- 12B dense maiprofile 已跑通(`docs/maiprofile_data_overview.md`):4096ctx block5 accept_len seasonality 5.88 / 其他 3.46–3.97;block7 更高但 verify_rate 降。
- DSpark 原生不支持 MoE:`deepspec/modeling/dspark/gemma4/modeling.py:175` `assert not config.enable_moe_block`。
- **vllm serve 依赖 PR #47216(未 merge,且明确 not MoE)**。

### 计划(从 dev/maiprofile 切 dev/maiprofile_moe,注意兼容性,后续 merge 回)
1. 研究 dense base↔draft 关系:draft 如何从 `target_layer_ids=[5,17,29,41,46]` 抽 hidden 对齐。
2. 借鉴 Google MTP 对 MoE 的处理(draft 保持 dense,挂在 MoE target 上)→ 启发:draft 可保持 dense,重点是正确抽 MoE target 的 hidden。
3. 给 `Gemma4DSparkDecoderLayer` 加 MoE 分支 / 或适配 MoE target hidden。
4. 训练 maiprofile 的 26B-A4B DSpark draft。
5. **测速受阻于 vLLM 支持**:先只记 accept rate 增长;等 #47216 merge 后先用 **12B** 验证 serve+bench 是否符合预期,再回到 MoE。

---

## 7. 优先级 & 依赖顺序

**P0(立即,解 blocker)**
- 账号 SSH 隔离(§1)。

**P1(并行启动,最快出对比数 / 重点线)**
- 方案1 Step A:官方 EAGLE-3 测速脚本(纯脚本,本地可写,服务器即跑,最快拿到第二个对比点)。
- 方案2 MTP:`debug_gemma_assistant.py` + 数据通路(重点线,耗时最长,尽早开工)。

**P2(数据/框架就绪后)**
- 方案1 Step B:EAGLE-3 maiprofile 训练。
- 方案3:DSpark MoE 结构改造 + 训练(先出 accept rate)。

**P3(受外部依赖)**
- DSpark serve+bench:**卡在 vLLM PR #47216 merge**。merge 后先 12B 验证,再 MoE。

---

## 8. Block 项 / 风险登记

| # | Block/风险 | 影响 | 缓解 |
|---|---|---|---|
| B1 | 账号 push 权限(SSH 未配) | 无法 push | §1,等用户加 key |
| B2 | vLLM PR #47216 未 merge 且 not MoE | DSpark(尤其 MoE)无法 serve+bench 实测速度 | 先记 accept rate;merge 后先 12B 验证;MoE 可能需自己改 vllm |
| B3 | MTP 训练需 target 的 shared_kv_states/inputs_embeds | 训练通路不通则方案2 停摆 | 先写 debug 脚本验证接口,再搭数据通路 |
| B4 | maiprofile 数据在 Azure ML mount | 本地无法访问数据 | 所有数据步骤只在服务器/容器跑,本地仅写代码 |
| B5 | MTP 产物 checkpoint 格式兼容性 | 训完无法被现有 vLLM 加载 | 导出对齐官方 assistant 格式,早验证 |
| B6 | DSpark MoE draft 与官方 MTP 结构差异 | 设计不当则加速不达标 | 参考 Google assistant「dense draft + MoE target」范式 |

---

## 9. 待确认(用户)

1. ✅ Git 组织:四 repo 分工(已定,见 §0)。
2. ✅ DSpark MoE 分支 `dev/maiprofile_moe`(已定)。
3. ✅ 镜像:用户容器内处理,我方只出脚本(已定)。
4. ✅ MTP 走选项 B 独立 repo(已定)。
5. ⬜ 优先级微调:P1 两条线(EAGLE-3 测速 + MTP 开发)先做哪个?还是同时?
6. ✅ 成功判据:**tok/s 和 accept_len 都要看**。tok/s 是最终目标;各 maiprofile 层的
   accept_len(以及 per-position acceptance)必须单独记录,用于定位每条线在不同层上的表现。
   所有测速/eval 脚本都要输出:总 tok/s + 逐层 accept_len + per-position acceptance。

---

## 附录 A — EAGLE-3 论文要点(arXiv:2503.01840,已精读)
- 核心:放弃 feature 预测,改**直接 token 预测** + **多层特征融合**(low/mid/high concat→FC→融合特征 g)。
- **TTT(training-time test)**:训练时模拟推理多步自回归,draft 输出 a 喂回自己继续训,适应"输入可能是 target 的 g 或自己的 a"。对应 DeepSpec `ttt_length`。
- 优势:去掉 feature 约束→表达力强、能吃更多数据、避免误差累积。最高 6.5x,比 EAGLE-2 快 ~1.4x。
- 训练:AdamW(0.9,0.95), grad clip 0.5, lr 5e-5, ShareGPT+UltraChat-200K。
- 指标:Speedup / Avg Accept Length τ / Accept Rate n-α(链式非树)。

## 附录 B — 本地已 clone 资料
| 资料 | 路径 | 分支 |
|---|---|---|
| vllm-msn | `~/workspace/vllm-msn` | feat/gemma4-12b-fp8-bench |
| DeepSpec | `~/workspace/DeepSpec` | dev/maiprofile |
| speculators fork | `~/workspace/speculators-fork` | main(待建 dev/maiprofile) |
| gemma4-mtp-trainer | `~/workspace/gemma4-mtp-trainer` | 空 |
| EAGLE-3 论文 | `/tmp/eagle3_paper.pdf` | v3 |

## 附录 C — vLLM PR #47216 情报(DSpark serve 支持)
- 标题:[Spec Decode][DSpark] Add Gemma4-12B DSpark draft model
- 状态:**OPEN, 未 merge**(mergeable: unstable),更新于 2026-07-08。
- 关键:Target `google/gemma-4-12B-it`,Draft `deepseek-ai/dspark_gemma4_12b_block7`。
- **明确 "mirrors the dense path, NOT the MoE/MLA one"** → 我们的 26B-A4B MoE draft 要 serve,大概率需在此 PR 基础上自己改 vllm。
