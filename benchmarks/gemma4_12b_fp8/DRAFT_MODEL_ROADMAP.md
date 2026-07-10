# Gemma4 Draft-Model 加速项目 — 总规划

> **最终目标(不变):26B-A4B MoE**(每 token 仅激活 ~4B) — 在 MTP 基线之上,
> 用 **MAI Profile 数据**训练更好的 draft,吞吐 +10%–20%。
> 逐层 MAI Profile 基线已实测(§2.5):26B-A4B MoE MTP 聚合 online **2678 tok/s** / accept_len **4.79** / accept **75.87%**(sc1_delta 聚合基线为 2113 / 5.03 / 80.58%)。
>
> **当前阶段:先在 dense Gemma4-12B 上跑出效果**,作为通往 MoE 的踏脚石 —— dense 社区支持成熟、
> 工具链无坑,先用它把「EAGLE-3 / DSpark + MAI Profile 训练」这套方法验证有效,再迁移回 26B-A4B MoE。
> dense-12B 阶段对标 **12B MTP 基线(逐层聚合 online 1382 tok/s / accept_len 4.46)**;最终阶段对标 26B-A4B MoE。
>
> 纪律:本地是开发机(无 GPU)。所有代码 **git commit(本地)→ 用户切好 SSH 后 push → 服务器/容器内 pull 出真实结果**。
> 本文档不编造任何数字;accept rate / tok/s 一律以服务器实测为准。镜像/环境由用户在容器内处理,我方只出脚本。

> **⚠️ 2026-07-09 路线调整(见 §10 变更记录)**
> - **speculators fork EAGLE-3 线已终止**(MoE hidden-states 抽取无解,详见 §4)。
> - **两阶段策略**:Phase 1 = DeepSpec EAGLE-3 + DSpark 在 **dense-12B** 验证方法(当前);Phase 2 = 迁回 **26B-A4B MoE**(最终目标,§6.5)。
> - MTP 线暂停(未终止,代码已就绪,见 §5);它是 Phase 2 attacking MoE 的候选路径之一。

---

## 0. 仓库地图(职责分离)

| Repo | 分支 | 职责 | 状态 |
|---|---|---|---|
| `zouxinyi0625-hue/vllm-msn` | `feat/gemma4-12b-fp8-bench` | 推理 / benchmark / 部署 / **本总规划文档** | 🟢 active,勿放训练代码 |
| `zouxinyi0625-hue/DeepSpec` | `dev/maiprofile` | **Phase 1 焦点**:EAGLE-3 + DSpark 训练(内含两套 modeling/trainer) | 🟢 active,dense 12B 跑通 |
| `zouxinyi0625-hue/gemma4-mtp-trainer` | `main` | MTP(自研),**Phase 2 攻 MoE 候选路径**,代码就绪未训 | ⏸ paused(§5) |
| `zouxinyi0625-hue/speculators`(fork) | `dev/maiprofile` | ~~方案 EAGLE-3(speculators 框架)~~ | ⛔ **已终止**(§4) |

---

## 1. 账号隔离(✅ 已解决,历史记录)

- **根因**:Copilot 鉴权跟 `gh` active 账号绑定。切到 `zouxinyi0625-hue`(无 Copilot 席位)→ 聊天 403。
- **方案**:`gh` 永远保持 **microsoft 账号 active**(保 Copilot);**push 走 SSH remote**(`git@github.com:zouxinyi0625-hue/...`,SSH 只认 key 绑定的账号,与 gh 登录无关)。两通道物理隔离。
- **状态**:✅ SSH key 已加、remote 已改,各 repo commit 正常 push。不再是 blocker。
- commit 署名:`Xinyi Zou <xinyizou@microsoft.com>` + `Assisted-by: Claude (Hermes Agent)` + `Signed-off-by:` trailer。

---

## 2. 当前真实基线(vllm-msn RESULTS.md)

**最终目标基线(26B-A4B)** = `26b_e011_mtp`, `google/gemma-4-26B-A4B-it` + 官方 assistant(MTP), `spec_tokens=5`。
**Phase 1 阶段基线(dense-12B)** = `12b_e011_mtp`, online **1133 tok/s**(accept_len 暂无记录)。

| 场景 | Output tok/s | accept length | accept rate |
|---|---:|---:|---:|
| offline `26b_e011_mtp` | **2023.06** | — | — |
| offline `26b_e011_no_mtp` | 1474.46 | — | — |
| **online `26b_e011_mtp`**(最终靶子) | **2113.78** | **5.03** | **80.58%** |
| **online 12B MTP**(Phase 1 靶子) | **1133.69** | — | — |

per-position acceptance(26B-A4B MTP):`pos0 93.61 / pos1 87.08 / pos2 80.45 / pos3 74.23 / pos4 67.54`
**达标线**:各阶段 +10~20% → dense-12B ~1246–1360 / 最终 26B-A4B 2325–2536。

### 官方 EAGLE-3 draft 已测速 —— net-negative(2026-07-08,已录 RESULTS.md)

部署 `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` + FP8 target,同 dataset/并发/`spec_tokens=5`:

| 指标 | 官方 EAGLE-3 | MTP 基线 | vs MTP |
|---|---:|---:|---:|
| online tok/s | **931** | 2113 | **−56%** |
| offline tok/s | 888 | 2023 | −56% |
| accept rate | **9.75%** | 80.58% | −70.8 pt |
| accept length | **1.49** | 5.03 | −3.54 |
| pos0 接受率 | 31% | 93.6% | — |

**结论**:通用 off-the-shelf draft 在 FP8 26B-A4B(MoE)+ MAI 分布上完全失效(甚至低于 no-MTP)。
→ **验证了项目核心命题:必须用 MAI Profile 数据自训 domain draft。** 这是所有训练线的出发点。

---

## 2.5 ✅ Per-Layer MAI Profile MTP 基线全部完成(2026-07-10,已录 RESULTS.md)

> **进展**:用 MAI Profile 5 个短层的 eval 数据(各 200 prompt),对 **12B(dense)** 和 **26B-A4B(MoE,最终目标)** 各测了 MTP + no-MTP 逐层 online 基线。
> 这套逐层 accept rate + 逐层 MTP 加速比,就是**自训 draft 要逐层超越的靶子**。全部服务器实测,零编造。

**5 层**:`layer1_actual` / `layer1_intent` / `layer2_temporal` / `layer3_seasonality` / `layer4_commercial_preference`
**驱动**:`run_maiprofile_online.sh`(12B)、`run_26b_maiprofile_online.sh`(26B 薄封装,同 driver);unlimited 并发,`spec_tokens=5`,temp 0.7。

### 聚合结果(wall-clock 加权)

| 模型 | 结构 | MTP accept rate | MTP accept_len | dense/no-MTP tok/s | MTP tok/s | **MTP 加速比** |
|---|---|---:|---:|---:|---:|---:|
| Gemma4-12B | dense | 69.28% | 4.46 | 1105.97 | 1382.32 | **1.25×** |
| **Gemma4-26B-A4B**(最终目标) | **MoE(~4B active)** | 75.87% | 4.79 | 1766.15 | 2678.67 | **1.52×** |

- **26B-A4B(MoE)MTP 每层 accept_len 都超 12B、tok/s ~1.9×**;MTP 对 MoE 帮助(1.52×)大于对 dense-12B(1.25×)。
- **逐层 accept rate(26B-A4B MoE MTP)**:seasonality **99.03%** / actual 75.45% / commercial 69.10% / intent 59.34% / temporal 54.50%。
- **逐层 MTP 加速比(26B-A4B MoE)**:seasonality **2.25×** / commercial 1.39× / intent 1.37× / temporal 1.31× / actual 1.27×。
- **共性**:seasonality 近饱和(谁训都容易,已近天花板);free-form 层(intent/temporal/actual/commercial)是真正有提升空间的地方 → **自训 draft 应主攻 free-form 层**。
- `layer1_actual` 反常(accept 高但加速比低):生成短、TTFT 主导,decode 端 MTP 收益被稀释。

### DSpark(自训 draft,block5,dense-12B)vs 12B MTP(同口径,已录 DeepSpec docs)
- 逐层 accept_len:DSpark 目前**整体略逊 12B MTP**(均值 4.18 vs 4.27);DSpark 强在尾部(pos4 在 temporal/commercial 反超),但首 token(pos0)MTP 几乎全赢。
- ⚠️ 口径:DSpark `verify_rate` ≠ MTP `accept rate`(定义不同),只 accept_len 可直接比。block7 需 MTP 也跑 spec=7 才公平对比。
- **这是 Phase 1 的关键读数**:方法在 dense-12B 上验证中,DSpark 尚未超 MTP,需继续调(block size / 训练步数 / 层加权)。

### 达标线(逐层化后)
- 最终 26B-A4B MoE:自训 draft 要在每层超过上面的 MTP 加速比,聚合 +10~20% over 2678 → **~2946–3214 tok/s**。
- Phase 1 dense-12B:自训 draft(DSpark/EAGLE-3)逐层 accept_len 超 12B MTP,tok/s 超 1382。

---

## 3. 三方式对比矩阵(更新后)

| 维度 | DeepSpec-EAGLE-3(Phase 1 焦点) | DeepSpec-DSpark(Phase 1 焦点) | MTP(Google assistant) | ~~speculators-EAGLE-3~~ |
|---|---|---|---|---|
| 代码位置 | `DeepSpec/deepspec/modeling/eagle3` | `DeepSpec/deepspec/modeling/dspark` | `gemma4-mtp-trainer`(自研) | ~~speculators fork~~ |
| **能训 MoE(26B-A4B)** | ❌ `assert not enable_moe_block` | ❌ 同左,写死 dense | ✅(=当前基线结构) | ✅ 框架支持 |
| **hidden-states 抽取** | ✅ 稳(DeepSpec target_cache) | ✅ 稳(同套 cache) | ⚠️ 需搭 shared_kv 通路 | ⛔ **MoE 上坏(§4)** |
| 能训 dense 12B | ✅ config 现成 | ✅ 已跑通 | — | ✅ |
| vllm serve 可用性 | ✅ 成熟 | ⚠️ PR #47216 未 merge 且仅 dense | ✅ 线上在用 | ✅ |
| 训练 loss | soft CE 蒸馏 + TTT(ttt=7) | CE .1 + L1/TV .9 + confidence | 自研蒸馏 | soft CE + TTT |
| **当前状态** | 🟢 Phase 1 焦点 | 🟢 Phase 1 焦点 | ⏸ paused(Phase 2 候选) | ⛔ 终止 |

> **关键洞察**:MoE(26B-A4B)在两个方向上都撞墙,且是同一障碍的两面 ——
> - **speculators**:modeling 支持 MoE,但 vLLM hidden-states connector 在 MoE 上抽取失败(§4)。
> - **DeepSpec**:hidden-states 抽取稳,但 modeling 写死 `assert not enable_moe_block`,拒绝 MoE。
>
> **两阶段策略** = Phase 1 先在 dense-12B(社区支持成熟、工具链无坑)把方法验证扎实(§6);
> Phase 2 迁回 26B-A4B MoE 最终目标(§6.5)。**dense-12B 是踏脚石,不是终点。**

---

## 4. ⛔ 已终止:speculators fork EAGLE-3(26B-A4B)

> **终止时间**:2026-07-09。分支 `speculators-fork@dev/maiprofile` 冻结,脚本保留供复盘,不再推进。

### 死因(三条,全部实测确认)
1. **hidden-states prepare 大量 mismatch**:vLLM 的 `ExampleHiddenStatesConnector` 对 Gemma4-26B-A4B(MoE)抽取,
   **online / offline API 都有** ~25–30% 样本 `hs len != tokens`(partial / len=0),被迫 skip。跑多少并发都一样。
2. **退回 transformers 又撞墙**:`flash_attn` 不支持 `head_dim > 256`(Gemma4 MoE 正是),
   不开 flash → SDPA fallback 到 math mode,~38s/it,慢到不可用。
3. **社区无参照**:这条 MoE + speculators 路线没人训过,无先例可循,试错成本过高。

> **注**:此终止**不影响最终目标**。26B-A4B MoE 仍是终点(§6.5),只是不再走 speculators 这条实现路径。

### 附:该线曾产出的资产(冻结在 `examples/train/maiprofile/`)
`split_maiprofile_eagle3.py` / `regenerate_maiprofile.py` / `prepare_maiprofile_eagle3.sh` /
`gen_hidden_states_{all.sh, vllm_offline.py, transformers.py}` / `launch_vllm_gemma4_26b.sh` /
`train_eagle3_maiprofile.sh` / `run_e2e.sh` / `probe_gemma4_26b.py`。
(regen ~73k ok、prepare ~69k valid 是真的;卡死在 hidden-states 这一步。)

---

## 5. ⏸ 暂停:MTP 训练(repo: gemma4-mtp-trainer)—— Phase 2 攻 MoE 候选路径

> 未终止,代码已就绪(`train.py` + 数据 pipeline + 单测,interface VERIFIED),Phase 1 集中到 DeepSpec 后暂停。

- 官方 draft = `Gemma4AssistantForCausalLM`(4 层 dense,hidden 1024,backbone 2816),
  forward 强依赖 target 的 `shared_kv_states` + `inputs_embeds`。
- 已搭:从 target 抽这些量的 debug 通路 + 单步蒸馏训练循环 + freeze 策略(已测)。
- **复活条件**:进入 Phase 2 攻 26B-A4B MoE 时,MTP 是唯一"结构原生支持 MoE 且已在线上验证"的路径 → 优先级回升(§6.5 路径 b)。

---

## 6. 🟢 Phase 1:DeepSpec EAGLE-3 + DSpark(dense Gemma4-12B)

> repo:`DeepSpec@dev/maiprofile`。两套 draft(EAGLE-3 / DSpark)共用同一份 MAI Profile target_cache。
> **目的:在社区支持成熟的 dense-12B 上把方法跑通、验证有效,为 Phase 2 迁 MoE 铺路。**

### 6.1 为什么能"几乎零改动"跑 EAGLE-3(代码确认)
1. **config 现成**:`config/eagle3/eagle3_gemma4_12b.py`
   (`trainer_cls=Gemma4Eagle3Trainer`,`target google/gemma-4-12B-it`,`ttt_length=7`,`draft_num_hidden_layers=1`,lr 6e-4)。
2. **cache 直接复用**:EAGLE-3 与 DSpark 的 `target_layer_ids` **完全一致 = `[5,17,29,41,46]`**,
   `target_cache_dataset.py` 的 layer-id assert 会通过 → **不用重抽数据**,DSpark 已有的 target_cache 直接喂。
3. **dense 不触发 MoE 墙**:12B 是 dense,不撞 `modeling/eagle3/gemma4/modeling.py:191` 的 `assert not enable_moe_block`。

### 6.2 训练命令(复用 DSpark sync 脚本,一行切 config)
```bash
cd ~/DeepSpec   # 服务器

CONFIG_PATH=config/eagle3/eagle3_gemma4_12b.py \
EXP_NAME=eagle3_ttt7_gemma4_12b_maiprofile \
bash scripts/train_maiprofile/train_dspark_short_sync.sh
```
自动接上 DSpark 同一 cache:`$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/target_cache/20260615/gemma4_12b_maiprofile_short_layers`。
调步数:加 `MAX_TRAIN_STEPS=3000 CHECKPOINTING_STEPS=250`(脚本已支持)。
> 提醒:cache 复用前提是启动时 layer-id assert 通过;若报 layer 不匹配 = 你那份 cache 抽取层不同,需重抽。本地无法预验(cache 在 mount 上)。

### 6.3 DSpark dense-12B 现状(已跑通,docs/maiprofile_data_overview.md)
- 4096ctx block5 各层 accept_len:seasonality **5.88** / temporal 3.96 / commercial 3.97 / actual 3.65 / intent 3.46。
- block7 更高(seasonality 7.76)但 verify_rate 降 ~0.1 → 块大小是端到端权衡,非单看 accept_len。
- 校准良好:ECE 0.5%(seasonality)~3–5%(其它),AUC 0.87–0.98。
- 教训:MAI prompt 长,max_length=1024 只有 27.9% 有效样本;提到 4096 → 99.2% 有效。**只信 4096 的 run。**

### 6.4 待办(Phase 1,dense-12B)
1. 跑 EAGLE-3 dense-12B maiprofile 训练(§6.2),出各层 accept_len + per-position + 校准。
2. 与 DSpark dense-12B 同口径对比(同 cache、同 block、同 eval),看哪套 draft 质量更高。
   **现状(2026-07-10)**:DSpark(block5)vs 12B MTP 已对比(§2.5 / DeepSpec docs)——DSpark 逐层 accept_len 均值 **4.18 尚未超** 12B MTP 的 **4.27**(尾部反超但首 token 输)。**方法在验证中,DSpark 还需调**(block size / 步数 / 层加权);EAGLE-3 dense-12B 训练+对比待补。
3. **端到端 tok/s 待 serve**:DSpark serve 卡 vLLM PR #47216(§8 B2);EAGLE-3 dense-12B 走成熟 serve 路径,可先出 tok/s,对标 12B MTP 逐层聚合 **1382 tok/s / accept_len 4.46**。
4. 记录:总 tok/s + 逐层 accept_len + per-position acceptance(§9 成功判据)。**12B/26B MTP + no-MTP 逐层基线已全部录入(§2.5),作为超越靶子。**
5. **Phase 1 出口判据**:确认 EAGLE-3/DSpark + MAI Profile 这套方法在 dense-12B 上确实 **逐层超过 12B MTP**(accept_len > 4.46 / tok/s > 1382),并选出更优的那套 draft → 带着这套方法进 Phase 2。

### 6.5 Phase 2:迁回 26B-A4B MoE(最终目标)
> **dense-12B 是踏脚石,最终必须回到 26B-A4B MoE。** 方法在 Phase 1 验证通过后启动。

已知 MoE 障碍与候选解:
- **障碍**:DeepSpec eagle3+dspark 的 `modeling/gemma4/modeling.py` 都 `assert not enable_moe_block`(dense-only)。
- **路径 a(改 DeepSpec)**:拆 assert,给 draft 正确对齐 MoE target 的 hidden ——
  draft 本身可保持 dense(参考 Google MTP「dense draft 挂 MoE target」范式),关键是抽/喂 MoE target hidden 的通路正确。
- **路径 b(MTP)**:复活 gemma4-mtp-trainer(§5),官方 assistant 结构原生支持 MoE、已是线上基线,
  导出对齐官方 assistant 格式 → 能被现有 `26b_e011_mtp` vLLM 直接加载。
- **serve 侧**:26B-A4B MoE draft 要 serve,需在 vLLM PR #47216(明确 not MoE)基础上自己改 vllm。
- **对标**:26B-A4B MoE MTP —— sc1_delta 聚合 online 2113 / accept_len 5.03;逐层 MAI Profile 聚合 **2678 tok/s / accept_len 4.79 / 加速比 1.52×**(§2.5)。自训 draft 目标 +10~20% over 2678 → **~2946–3214 tok/s**,且逐层加速比要超 MTP(尤其 free-form 层)。

---

## 7. 优先级 & 依赖顺序(两阶段)

### Phase 1 — dense Gemma4-12B 验证方法(当前)
- **P0** — DeepSpec **EAGLE-3 dense-12B** maiprofile 训练 + eval(§6.2),与 DSpark dense-12B 同口径对比(同 cache/block/eval)。
- **P1** — EAGLE-3 dense-12B 走成熟 vLLM serve 出**真实 tok/s**(不卡 PR,最快端到端对比点),对标 12B MTP(1133)。
- **P1** — DSpark dense-12B serve:等 PR #47216 merge(§8 B2)。
- **出口** — 方法验证通过 + 选出更优 draft → 进 Phase 2。

### Phase 2 — 迁回 26B-A4B MoE(最终目标)
- **P2** — 把 Phase 1 验证好的方法搬到 MoE,两条候选路径(§6.5):
  (a) 改 DeepSpec:拆 `assert not enable_moe_block` + 正确处理 MoE target hidden;
  (b) 复活 MTP 线(gemma4-mtp-trainer,§5),结构原生支持 MoE 且已线上验证。
- **P3** — MoE serve+bench:DSpark 卡 PR #47216(明确 not MoE),需在其基础上改 vllm;对标 26B-A4B MTP(2113)。

---

## 8. Block 项 / 风险登记

| # | Block/风险 | 影响 | 缓解 / 状态 |
|---|---|---|---|
| ~~B1~~ | ~~账号 push 权限(SSH)~~ | — | ✅ 已解决(§1) |
| B2 | vLLM PR #47216 未 merge 且 not MoE | DSpark(尤其 MoE)无法 serve+bench 实测速度 | 先记 accept rate;EAGLE-3 dense 走成熟 serve 先出 tok/s;merge 后先 12B 验证 |
| **B3** | **MoE hidden-states 双向撞墙** | 26B-A4B(最终目标)两条路都撞墙(§3 洞察) | Phase 1 先锁 dense-12B 验证方法;Phase 2 攻 MoE(§6.5 路径 a/b) |
| B4 | maiprofile 数据在 Azure ML mount | 本地无法访问数据/cache | 所有数据步骤只在服务器/容器跑,本地仅写代码 |
| B5 | EAGLE-3 cache 复用需 layer-id assert 通过 | assert 失败则需重抽 cache | layer_ids 两边一致 = `[5,17,29,41,46]`,理论通过;本地无法预验,报错即贴 |
| B6 | draft 质量高 ≠ 端到端快 | accept_len 好但 tok/s 可能不达标 | 坚持两个都测,tok/s 以 serve 实测为准,不用 accept rate 冒充加速比 |
| **B7** | **vllm-msn 无 Gemma4 EAGLE-3 draft 模型类** | DeepSeek release draft(`deepseek-ai/eagle3_gemma4_12b_ttt7`,arch `Gemma4Eagle3Model`)无法在 vllm-msn serve:`registry.py` 只注册了 Gemma4 的 target(`Gemma4ForCausalLM`)+ MTP(`Gemma4MTPModel`),**无 eagle3**;`gemma4.py` 也无 `Gemma4Eagle3Model` 类。启动即 pydantic ValidationError | 选项:(a) 用 upstream/deepseek 官方支持该 arch 的 vLLM 版本测 draft 效果;(b) 照 `llama_eagle3.py::Eagle3LlamaForCausalLM`(~110 行)移植 Gemma4 draft 类 + 注册;(c) 先看 draft config.json 确认是否只是 arch 名没映射。**未决,待定方向** |

---

## 9. 成功判据(用户确认)

- **tok/s 和 accept_len 都要看**。tok/s 是最终目标;各 maiprofile 层的 accept_len(及 per-position acceptance)必须单独记录。
- 所有测速/eval 脚本都要输出:**总 tok/s + 逐层 accept_len + per-position acceptance**。
- 达标线(分阶段):Phase 1 dense-12B +10~20% over 12B MTP(1133)→ ~1246–1360;最终 26B-A4B +10~20% over 2113 → 2325–2536。

---

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-07-07 | 初版:三条线并行(EAGLE-3 / MTP / DSpark),四 repo 分工,账号 blocker。 |
| 2026-07-08 | 官方 EAGLE-3 draft 测速完成 → net-negative(931 tok/s / 9.75%),验证需自训。 |
| 2026-07-08 | 账号 SSH 隔离生效,B1 解除。 |
| 2026-07-09 | **路线调整 + 两阶段确立**:speculators fork EAGLE-3(26B-A4B)因 MoE hidden-states 双向撞墙**终止**(§4)。确立两阶段:Phase 1 用 **dense-12B**(社区支持成熟)验证 EAGLE-3/DSpark+MAI 方法;Phase 2 **迁回 26B-A4B MoE(最终目标,不放弃)**(§6.5)。MTP 暂停,作为 Phase 2 候选路径。 |
| 2026-07-10 | **逐层 MAI Profile MTP 基线全部完成(§2.5)**:12B(dense)+ 26B-A4B(MoE)各测 MTP + no-MTP 逐层 online。26B-A4B MoE MTP 聚合 2678 tok/s / accept_len 4.79 / 加速比 1.52×(seasonality 2.25×,free-form 层 1.27–1.39×)。DSpark(自训 block5,dense-12B)逐层 accept_len 均值 4.18,**尚未超** 12B MTP(4.27),需继续调。新增 `run_26b_maiprofile_online.sh`(薄封装,统一走 `GEMMA4_MODEL_PATH`,修了 12B/26B 路径串味导致的 CUDA gather 崩溃)。 |

---

## 附录 A — EAGLE-3 论文要点(arXiv:2503.01840)
- 核心:放弃 feature 预测,改**直接 token 预测** + **多层特征融合**(low/mid/high concat→FC→融合特征 g)。
- **TTT**:训练时模拟推理多步自回归,draft 输出 a 喂回自己继续训。对应 DeepSpec `ttt_length`(config 里 =7)。
- 训练:AdamW(0.9,0.95), grad clip 0.5, ShareGPT+UltraChat-200K。指标:Speedup / Avg Accept Length τ / Accept Rate n-α。

## 附录 B — 本地已 clone 资料
| 资料 | 路径 | 分支 |
|---|---|---|
| vllm-msn | `~/workspace/vllm-msn` | feat/gemma4-12b-fp8-bench |
| DeepSpec(Phase 1 焦点) | `~/workspace/DeepSpec` | dev/maiprofile |
| gemma4-mtp-trainer(Phase 2 候选) | `~/workspace/gemma4-mtp-trainer` | main |
| speculators fork(终止) | `~/workspace/speculators-fork` | dev/maiprofile(冻结) |

## 附录 C — vLLM PR #47216 情报(DSpark serve 支持)
- 标题:[Spec Decode][DSpark] Add Gemma4-12B DSpark draft model。状态:**OPEN, 未 merge**(更新于 2026-07-08)。
- Target `google/gemma-4-12B-it`,Draft `deepseek-ai/dspark_gemma4_12b_block7`。
- **明确 "mirrors the dense path, NOT the MoE/MLA one"** → 26B-A4B MoE draft 要 serve 需在此 PR 基础上自己改 vllm。
