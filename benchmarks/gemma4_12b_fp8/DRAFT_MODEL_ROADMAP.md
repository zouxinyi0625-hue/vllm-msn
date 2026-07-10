# Gemma4 Draft-Model 加速项目 — 总规划

> **一句话目标**:给 **Gemma4-26B-A4B(MoE)** 训一个用 **MAI Profile 数据**的 draft model,
> 在 MTP 基线之上把推理吞吐再 **+10~20%**(→ ~2946–3214 tok/s)。
>
> **纪律**:本地开发机无 GPU。代码 commit(本地)→ push(SSH personal 账号)→ 服务器 pull 出真实结果。
> **不编造任何数字**;accept rate / tok/s 一律服务器实测。数据/cache 在 Azure ML mount,本地只写代码。

**最后更新:2026-07-10** · 详细数据见 `RESULTS.md`(vllm-msn)与 `docs/maiprofile_data_overview.md`(DeepSpec)。

---

## ★ 一、仓库地图（可跳转）

| Repo | 分支 | 职责 | 状态 |
|---|---|---|---|
| [`vllm-msn`](https://github.com/zouxinyi0625-hue/vllm-msn/tree/feat/gemma4-12b-fp8-bench/benchmarks/gemma4_12b_fp8) | `feat/gemma4-12b-fp8-bench` | 推理 / benchmark / **本文档 + RESULTS.md** | 🟢 |
| [`DeepSpec`](https://github.com/zouxinyi0625-hue/DeepSpec/tree/dev/maiprofile) | `dev/maiprofile` | DSpark + EAGLE-3 训练（线 1 / 3b） | 🟢 |
| [`speculators-fork`](https://github.com/zouxinyi0625-hue/speculators/tree/dev/maiprofile) | `dev/maiprofile` | EAGLE-3 @ speculators（线 3a） | 🟢 复活 |
| [`gemma4-mtp-trainer`](https://github.com/zouxinyi0625-hue/gemma4-mtp-trainer) | `main` | MTP finetune 自研（线 2） | 🟡 未实训 |

commit 署名：`Xinyi Zou <xinyizou@microsoft.com>` + `Assisted-by: Claude (Hermes Agent)` + `Signed-off-by:`。

---

## ★ 二、项目主表：要做什么 / 现状 / 结论

> 三种自训 draft 方案并行。EAGLE-3 按训练框架分两支（DSpark / speculators）。
> **吞吐口径**：online `vllm bench serve`,MAI Profile 5 层聚合,`spec_tokens=5`,unlimited 并发,tok/s。
> **无 draft 参照(no-MTP baseline)**:12B dense **1106 tok/s** · 26B-A4B MoE **1766 tok/s**(自训 draft 至少要超过它才有意义)。

| 工作线 | 模型 | 原生训练支持 | 预训练模型 & 其 MAI 吞吐（vs no-MTP baseline） | 训练完成 | 训练后模型吞吐 / 卡点 |
|---|---|---|---|---|---|
| **—基线—** | 12B dense | — | no-MTP(无 draft) **1106** · Google MTP **1382** | — | 参照线 |
| **—基线—** | 26B-A4B MoE | — | no-MTP(无 draft) **1766** · Google MTP **2678** | — | 参照线（最终靶子） |
| **DSpark** | 12B dense | ✅（DeepSpec） | 有（`deepseek/dspark_gemma4_12b_block7`）· **没测吞吐**（vLLM 不能 serve，只有 accept_len） | ✅ block5/block7 | 🔴 **blocked**：vLLM 不支持 serve（PR #47216）→ 只有 accept_len（block5 均值 4.18） |
| **DSpark** | 26B-A4B MoE | ❌ `assert not enable_moe_block` | 无 | ❌ | 🛠 **待开发**：拆 assert + 对齐 MoE target hidden（§五路径a） |
| **MTP** | 12B dense | 🟡 无社区脚本，自研 single-step（缺 TTT） | 有（Google assistant） · **1382 tok/s**（1.25× over 1106） | ❌ 未实训 | ⏳ **待训**：脚本就绪未跑，需先补 TTT |
| **MTP** | 26B-A4B MoE | 🟡 自研，官方 assistant 原生支持 MoE | 有（Google assistant） · **2678 tok/s**（1.52× over 1766） | ❌ 未实训 | ⏳ **待训**：目标 +10~20% → ~2946–3214 tok/s |
| **EAGLE-3 @DSpark** | 12B dense | ✅（DeepSpec） | 有（`deepseek/eagle3_gemma4_12b_ttt7`） · **没测**（vLLM load 不了） | 🟡 2000 step loss 降 | 🔴 **eval 精度差待排查** + vLLM 无法 load（arch 未注册,B7） |
| **EAGLE-3 @DSpark** | 26B-A4B MoE | ❌ `assert not enable_moe_block`（同 DSpark MoE） | 无 | ❌ | 🛠 **待开发**：拆 assert + 对齐 MoE target hidden（§五路径a） |
| **EAGLE-3 @spec** | 26B-A4B MoE | ✅（speculators） | 有（`RedHatAI/…speculator.eagle3`） · **931 tok/s（负优化,<1766 baseline）** | 🟡 **进行中,周一(07-13)训完** | ⏳ hidden_states 07-10 修好,待训完 eval |

> **注**：EAGLE-3 @speculators **原生支持 26B MoE,故不测 12B**（直接攻最终目标）;12B 那格由 EAGLE-3 @DSpark 覆盖。

**三条核心结论**：
1. **自训是必须的** —— 官方 EAGLE-3 draft 在 26B MoE 上 net-negative（931 tok/s / accept 9.75%,比不开 spec 还慢）。
2. **DSpark 方法 work** —— 12B dense 训练后 accept rate 较未训练明显提升。目前最强正向信号。
3. **MoE 是分水岭** —— DSpark/DeepSpec-EAGLE3 写死 dense；只有 **MTP** 和 **speculators-EAGLE3** 能直接在 MoE 上训 → 最终目标靠这两条,或给 DSpark 加 MoE 支持。

---

## ★ 三、基线:自训 draft 要超越的靶子(全部实测,2026-07-10)

MAI Profile 5 短层(各 200 prompt),online `vllm bench serve`,`spec_tokens=5`,unlimited 并发。

### 聚合基线(wall-clock 加权)

| 模型 | 结构 | MTP accept | MTP accept_len | no-MTP tok/s | MTP tok/s | MTP 加速比 |
|---|---|---:|---:|---:|---:|---:|
| Gemma4-12B | dense | 69.28% | 4.46 | 1105.97 | 1382.32 | **1.25×** |
| **Gemma4-26B-A4B** | **MoE(~4B active)** | 75.87% | 4.79 | 1766.15 | 2678.67 | **1.52×** |

### 逐层 MTP(26B-A4B MoE)—— 自训 draft 要逐层超这条线

| Layer | accept | accept_len | 加速比 | 空间 |
|---|---:|---:|---:|---|
| layer3_seasonality | 99.03% | 5.95 | 2.25× | 近饱和,天花板 |
| layer1_actual | 75.45% | 4.77 | 1.27× | 短生成 TTFT-bound |
| layer4_commercial_preference | 69.10% | 4.46 | 1.39× | **有空间** |
| layer1_intent | 59.34% | 3.97 | 1.37× | **有空间** |
| layer2_temporal | 54.50% | 3.72 | 1.31× | **有空间** |

**要点**:seasonality 已近天花板(谁训都容易);**自训 draft 应主攻 free-form 层**(intent/temporal/commercial/actual)。

### 达标线
- **最终(26B-A4B MoE)**:自训 draft 聚合 +10~20% over 2678 → **~2946–3214 tok/s**,逐层加速比超 MTP。
- **Phase 1(dense-12B 验证)**:逐层 accept_len 超 12B MTP(4.46),tok/s 超 1382。

### 官方 EAGLE-3 draft(26B MoE)已测 → 负优化(证明必须自训)
`RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`,同 dataset/并发:online **931 tok/s**(−56% vs MTP 2113)、
accept **9.75%**、accept_len **1.49**、pos0 31%。**通用 off-the-shelf draft 在 MAI 分布上完全失效。**

---

## ★ 四、当前重点 & 待办(按优先级)

| 优先级 | 事项 | 线 | 依赖 |
|---|---|---|---|
| **P0** | 等 speculators-EAGLE3 周一训完 → eval 逐层 accept | 3a | hidden_states(已修) |
| **P0** | 排查 DSpark-EAGLE3「loss 降但 eval 差」 | 3b | — |
| **P1** | DSpark serve:等 vLLM PR #47216 merge,先 12B 验证端到端 tok/s | 1 | PR #47216 |
| **P1** | 测新版 vLLM 能否 load DSpark-EAGLE3 模型 | 3b | — |
| **P2** | DSpark 继续调(block size / 步数 / 层加权,主攻 free-form 层) | 1 | — |
| **P2** | MTP finetune:补 TTT 多步自回归后跑一版 smoke | 2 | 改 training_step |
| **P3** | DSpark/DeepSpec-EAGLE3 迁 MoE:拆 `assert not enable_moe_block` + 对齐 MoE hidden | 1/3b | Phase 1 验证后 |

**成功判据**:tok/s 和 accept_len 都看;所有 eval 输出**总 tok/s + 逐层 accept_len + per-position acceptance**。
tok/s 以 serve 实测为准,**不用 accept rate 冒充加速比**(draft 质量高 ≠ 端到端快)。

---

## 五、各线技术细节

### 线 1 — DSpark(主力,repo `DeepSpec@dev/maiprofile`)
- **实测**:12B dense block5 逐层 accept_len:seasonality 5.88 / commercial 3.97 / temporal 3.96 / actual 3.65 / intent 3.46。block7 更高(seasonality 7.76)但 verify_rate 降 ~0.1 → 块大小是端到端权衡。
- **DSpark vs Google MTP**(同口径,DeepSpec docs):DSpark block5 accept_len 均值 4.18,略低于 12B MTP 4.27(尾部反超、首 token 输)。⚠️ DSpark `verify_rate` ≠ MTP `accept rate`,只 accept_len 可直接比。
- **教训**:MAI prompt 长,cache `max_length=1024` 仅 27.9% 有效样本;提到 **4096 → 99.2% 有效**。只信 4096 的 run。
- **限制**:① 无 MoE 原生训练(`modeling/gemma4/modeling.py` 写死 dense);② vLLM 不能 serve(PR #47216 未 merge 且仅 dense)。

### 线 2 — MTP finetune(repo `gemma4-mtp-trainer@main`)
- Google 只提供预训练 assistant(`Gemma4AssistantForCausalLM`,4 层 dense,hidden 1024,backbone 2816),forward 依赖 target 的 `shared_kv_states` + `inputs_embeds`。
- 自研:target 抽取通路 + 训练循环 + freeze 策略(冻结 tied lm_head/embed,只训 4 层 decoder + projection),单测过、接口 VERIFIED,**未实训**。
- **已知缺陷**:`training_step.py` 是 single-step 蒸馏,**无 TTT**,推理多步时尾部接受率会塌(推理连喂 5 步自己的输出,训练却只 teacher-forcing 1 步)。**推进前需补 TTT。**
- **战略价值**:唯一"结构原生支持 MoE 且已线上验证"的路径 → 可直接攻 26B-A4B。

### 线 3a — EAGLE-3 @ speculators(repo `speculators-fork@dev/maiprofile`)
- speculators 提供 26B MoE 预训练 EAGLE-3,但 MAI 上负优化 → 需自训。
- **2026-07-09 曾受阻**:vLLM hidden-states connector 对 26B MoE 抽取 ~25-30% 样本 mismatch(partial/len=0)。
- **✅ 2026-07-10 已解**:加 `--no-enable-prefix-caching`(prefix caching 是根因)。正在保存 hidden_states,**周一(07-13)训完**。
- 资产(`examples/train/maiprofile/`):`split/regenerate/prepare_maiprofile*` / `gen_hidden_states_*` / `train_eagle3_maiprofile.sh` / `run_e2e.sh`。regen ~73k ok、prepare ~69k valid。

### 线 3b — EAGLE-3 @ DSpark(repo `DeepSpec@dev/maiprofile`)
- config 现成:`config/eagle3/eagle3_gemma4_12b.py`(`ttt_length=7`,`draft_num_hidden_layers=1`,lr 6e-4)。
- cache 与 DSpark 共用:`target_layer_ids` 一致 `[5,17,29,41,46]`,不用重抽。
- **训练命令**:
  ```bash
  cd ~/DeepSpec
  CONFIG_PATH=config/eagle3/eagle3_gemma4_12b.py \
  EXP_NAME=eagle3_ttt7_gemma4_12b_maiprofile \
  bash scripts/train_maiprofile/train_dspark_short_sync.sh
  # 调步数: MAX_TRAIN_STEPS=3000 CHECKPOINTING_STEPS=250
  ```
- **现状**:训起来了,2000 step loss 降,**但 eval 精度差,原因待排查**(train/eval 口径?TTT?checkpoint 加载?)。
- **serve 问题**:DSpark 放出的 EAGLE-3 模型 **vLLM 无法 load**(arch `Gemma4Eagle3Model` 未在 vllm-msn `registry.py` 注册,`gemma4.py` 无该类)→ 待测新版 vLLM 是否支持。

### 迁 MoE 的两条候选路径(Phase 2)
- **路径 a(改 DeepSpec)**:拆 `assert not enable_moe_block`,给 draft 正确对齐 MoE target hidden(draft 本身可保持 dense,参考 MTP「dense draft 挂 MoE target」范式)。
- **路径 b(MTP)**:MTP 官方 assistant 原生支持 MoE、已线上验证,导出对齐格式即可被 `26b_e011_mtp` vLLM 直接加载。
- **serve 侧**:26B MoE draft serve 需在 vLLM PR #47216(明确 not MoE)基础上自己改 vllm。

---

## 六、Block / 风险

| # | 风险 | 影响 | 状态 |
|---|---|---|---|
| B2 | vLLM PR #47216 未 merge 且 not MoE | DSpark 无法 serve 出端到端 tok/s | 等 merge;先测 accept rate |
| B3 | DSpark/DeepSpec-EAGLE3 写死 dense | 最终 MoE 目标要改代码 | Phase 2 路径 a/b |
| B4 | 数据/cache 在 Azure ML mount | 本地无法访问 | 数据步骤只在服务器跑 |
| B6 | draft 质量高 ≠ 端到端快 | accept_len 好但 tok/s 未必达标 | 两个都测,tok/s 以 serve 为准 |
| B7 | vllm-msn 无 Gemma4 EAGLE-3 draft 类 | DSpark-EAGLE3 模型 load 不了 | 待测新版 vLLM;或移植 `Gemma4Eagle3Model`(照 `llama_eagle3.py` ~110 行) |

---

## 七、变更记录

| 日期 | 变更 |
|---|---|
| 2026-07-07 | 初版:三线并行(EAGLE-3 / MTP / DSpark)。 |
| 2026-07-08 | 官方 EAGLE-3 draft 测速 → 负优化(931 tok/s / 9.75%),验证需自训。 |
| 2026-07-09 | 曾判 speculators-EAGLE3 因 MoE hidden-states mismatch 终止(后于 07-10 复活)。 |
| 2026-07-10 | **逐层 MAI Profile 基线全部完成**(12B + 26B-A4B MTP/no-MTP,§二)。26B MoE MTP 聚合 2678 tok/s / 加速比 1.52×。新增 `run_26b_maiprofile_online.sh`,修了 12B/26B 路径串味导致的 CUDA gather 崩溃。 |
| 2026-07-10 | **三线状态更新**:DSpark 12B 训练有明显增益(主力);MTP 探索(无 finetune 先例、缺 TTT);**speculators-EAGLE3 复活**(prefix-caching bug 修好,周一训完);DSpark-EAGLE3 loss 降但 eval 差。**认知修正**:speculators-EAGLE3 与 MTP 能直接在 MoE 上训,三线并进。 |

---

## 附录 — EAGLE-3 论文要点(arXiv:2503.01840)
- 核心:放弃 feature 预测,改**直接 token 预测** + **多层特征融合**(low/mid/high concat→FC→融合特征 g)。
- **TTT**:训练时模拟推理多步自回归,draft 输出喂回自己继续训(DeepSpec `ttt_length=7`)。这是尾部接受率不塌的关键。
- 训练:AdamW(0.9,0.95),grad clip 0.5。指标:Speedup / Avg Accept Length τ / Accept Rate n-α。

## 附录 — vLLM PR #47216(DSpark serve)
- [Spec Decode][DSpark] Add Gemma4-12B DSpark draft model。**OPEN,未 merge**。
- Target `google/gemma-4-12B-it`,Draft `deepseek-ai/dspark_gemma4_12b_block7`。
- 明确 "mirrors the dense path, NOT the MoE/MLA one" → 26B MoE serve 需在此基础上改 vllm。
