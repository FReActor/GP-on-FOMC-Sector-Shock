# SFIC FA26 项目总览：用 Genetic Programming 预测美联储利率冲击下的行业反应

> 更新于 2026-11-15（Stage 1 results freeze，tag `stage1-freeze`）。作者：Charles Cheng。
> 这份文档给团队成员、leads 和以后接手的人看：项目背景、完整 workflow、每个人做了什么、还剩什么。
> 数字都来自 `results/summary.csv`，图在 `results/figures/`。

---

## 1. 项目背景

### 1.1 研究问题

每次 FOMC（美联储议息会议）宣布利率决定时，如果结果和市场预期不一样（"surprise"），股市会立刻反应，而且**不同行业的反应不一样**：比如加息意外时，utilities、real estate 这类利率敏感行业通常跌得更多。

我们的问题是：

> **能不能用 Genetic Programming (GP) 从数据里自动找出一个"公式"，在 FOMC 宣布前就预测每个行业相对大盘的反应？这个公式能不能比简单的线性模型更好？**

GP（也叫 symbolic regression）的卖点是：它不是黑箱，而是直接输出一个可读的数学公式（例如 `0.162 × beta_1y`），所以如果真有非线性结构，我们可以看见并解释它。

### 1.2 为什么值得做（文献）

- GP 在金融里的结果**有正有负**：Neely et al. (1997) 用 GP 找到外汇交易规则有效；Allen & Karjalainen (1999) 用 GP 找股票交易规则，扣除成本后没有超额收益。
- Chen & Yeh (2000)、Stasinakis et al. (2013) 把 GP / symbolic regression 用在金融和宏观预测上。
- 事件研究这边，Bernanke & Kuttner (2005) 是经典：用利率期货算 Fed surprise，看股市怎么反应。
- 我们的切入点：把 GP 用在**小样本的事件研究**（每个行业只有约 200 个 FOMC 事件）上，看它能找到真结构，还是只会 overfit。

### 1.3 两个阶段

| | Stage 1：社团展示 | Stage 2：论文 |
|---|---|---|
| 时间 | 2026-11-03 到 11-22（周末展示） | 2026-12 到 2027-01（GECCO 2027：abstract 1/18，paper 1/25） |
| 数据 | 2 年期国债收益率当天变化当 surprise；9 个 SPDR 行业 ETF；1999-2025 | SF Fed 官方 Bauer-Swanson surprise（30 分钟窗口）；Ken French 49 个行业；1994-2025 |
| 模型 | gplearn GP vs 线性回归 vs zero forecast | PySR + gplearn vs 多个 baseline，多 seed，显著性检验，placebo |
| 测试集 | 2010-2015（2016 以后留给论文，不碰） | 锁定的 2016+ |

本文档主要讲 **Stage 1**，Stage 2 的情况在第 6 节。

---

## 2. 团队

| 人 | 角色 |
|---|---|
| Razeen Khan、Benny Signer | Leads：review、pre-registration、论文 intro / related work、advisor 联络 |
| **Charles Cheng** | Person 1：GP + evaluation，Stage 1 的 critical path（数据管线开头、所有 GP pooled 跑、baseline、对比、结果、图、slides 合成） |
| Abhishai (Abhi) Ganta | Person 3：features、tests/CI、data report，后来加入 GP（per-ticker v1、公式解读），现在主做 Stage 2 |
| Snata Mohanty | Person 2：slides（Motivation + literature、Baseline），Colab 初步探索 |

---

## 3. 完整 workflow

### 3.1 数据管线

```
data/reference/fomc_dates.csv          ← 222 次 FOMC 宣布（1999-02 到 2025-12；215 次例会 + 7 次紧急降息）
        │
python -m src.fetch_data               → data/raw/prices.csv（SPY + 11 个 SPDR ETF，Yahoo）
                                         data/raw/fred.csv（DGS2、DGS10、VIX、BAA10Y、DFF，FRED）
python -m src.build_events             → data/processed/events.csv
                                         surprise_bp = 2y 收益率从 day −1 收盘到 day 0 收盘的变化（bp）
python -m src.build_returns            → data/processed/returns.csv
                                         excess_1d / excess_5d = 行业 ETF 收益 − SPY 收益
python -m src.build_features           → data/processed/features.csv
                                         day −1 的 vix、slope_2s10s、beta_1y、mom_20d（无 look-ahead）
```

`data/` 不进 git，每个人本地按上面顺序重跑。`src/data.py` 的 `load_panel()` 把 events、returns、features 拼成一个 panel，并按 `config.SPLIT` 标出 train / val / test。

### 3.2 关键定义（全部在 `src/config.py`）

- **Split A**：train 1999-2009（其中 2007-2009 是 validation，用来选 GP 的 parsimony），**test 2010-2015**。
- **Tickers**：9 个原始 SPDR 行业（XLB XLE XLF XLI XLK XLP XLU XLV XLY）。XLRE、XLC 2016 前几乎没数据。
- **Horizons**：1d、5d。
- **Metric**：`r2_oos = 1 − Σ(y − ŷ)² / Σ(y − ȳ_train)²`，ȳ_train 是该行业 1999-2009 的平均反应。同时报告 rmse 和 hit rate。
- **Seeds**：GP 每个配置跑 10 个 seed，报告中位数。

### 3.3 模型

| 模型 | 文件 | 输入 |
|---|---|---|
| zero forecast | `src/compare.py`（参考行） | 无，永远预测 0 |
| train_mean | `src/compare.py`（参考行） | 行业训练期均值（r2_oos 定义上 = 0） |
| linear | `src/models/linear.py` | surprise_bp |
| linear_interact | `src/models/linear_interact.py` | 4 个 features + surprise × vix / slope / 1{s>0} |
| GP v0 | `src/models/gp.py --version v0` | surprise_bp |
| GP v1 | `src/models/gp.py --version v1` | surprise_bp + 4 个 features（标准化） |

GP 设置（`config.GP`）：population 1000、40 代、tournament 20、函数 `add sub mul div max min abs neg`、parsimony 从 {0.0005, 0.001, 0.005, 0.01} 里在 validation 年份上选。两种 scope：**pooled**（9 个行业一个模型，主结果）和 **per-ticker**（每个行业一个模型，约 93 个训练事件）。

### 3.4 评估与出图

```
python -m src.models.linear
python -m src.models.linear_interact
python -m src.models.gp --version v0
python -m src.models.gp --version v1 --scope pooled      # Charles 的机器
python -m src.models.gp --version v1 --scope tickers     # Abhi 的机器
python -m src.compare        → results/summary.csv + results/go_no_go.md
python -m src.interpret      → results/gp_interpretation.md, gp_feature_usage.csv
python -m src.plots          → results/figures/*.png, results/gp_overfit.csv
pytest                       → 29 个测试（features、returns、GP 工具）
```

**Go / no-go 规则（看结果之前写好的）**：pooled 1d 上，GP v1 的中位数 r2_oos 要同时超过 linear 和 linear_interact，且 10 个 seed 里至少 6 个超过 linear，才算 GO。

---

## 4. Stage 1 结果（已 freeze）

### 4.1 主结果：没有模型比"永远预测 0"更好

Pooled，test 2010-2015（`results/figures/fig1_pooled_r2.png`）：

| 模型 | 1d | 5d |
|---|---|---|
| zero forecast | **+0.080** | **+0.010** |
| GP v0 | +0.080 | +0.004 |
| linear | +0.076 | +0.010 |
| GP v1 | +0.074 | +0.008 |
| linear_interact | +0.056 | +0.000 |

- **Go / no-go：NO-GO**。只有 1/10 个 seed 超过 linear。
- Pooled GP v0 的 20 个 seed-run 全部输出常数：GP 自己判断 surprise 在 pooled 情况下没有可用信号。
- Per-ticker：GP v1 只在 18 个 (行业, horizon) 格子里的 4 个超过 zero，和随机差不多；在 XLB 上崩了（1d −0.43，5d −0.36）。

### 4.2 GP 在 overfit

- 167 个用到了输入变量的 per-ticker GP seed-run 里，**76% 在 test 上 r2_oos < 0**，包括训练期 r2 有 0.3-0.55 的那些（`fig3_overfit.png`）。
- 典型例子（`fig4_xle_formula.png`）：XLE 1d 的公式 `Min(0.022/s², 0.0121/s)`。s ≠ 0 时几乎就是 0；s = 0 时 gplearn 的 protected division 返回 1，公式就预测 −1%。它只是拟合了训练期 3 个 s = 0 的日子；2010-2015 零利率时期有 8 个 s = 0 的事件，test r2_oos = −0.41。

### 4.3 一个陷阱：+0.08 不是能力

r2_oos 是和"行业训练期均值"比。1999-2009 年 FOMC 日的行业均值很大（XLU −49 bp、XLP −43 bp、XLF +32 bp），到 2010-2015 缩小了很多（`fig5_zero_artifact.png`）。所以**什么都不预测的 zero forecast 就已经有 +0.08**，单个行业甚至 +0.25。所有模型都必须和 zero forecast 比，而不是和 0 比。这一点会带进 Stage 2。

### 4.4 Track C：加 features 也没用

GP v1 确实用了 features（5d 时 mom_20d、beta_1y 出现在一半以上的公式里，`fig6_feature_usage.png`），但 test 表现没变好；linear_interact 比 linear 还差。作为 negative result 保留。

### 4.5 结论

只用 FOMC 当天 2y 收益率变化做 surprise，行业超额收益在样本外**无法预测**；GP 在小样本事件研究里很容易 overfit。下一步的关键是更好的 surprise 指标和更多数据，而不是更灵活的模型。Slides 文字和 speaker notes：`docs/slides_results.md`；书面版：`docs/stage1_writeup.md` 第 4、5 节。

---

## 5. 每个人做了什么

以下根据 git 历史（`git log --all`）和 `docs/STAGE1_PLAN.md` 整理。

### 5.1 Charles Cheng（Person 1，critical path）

| 日期 | 工作 | Commit / 文件 |
|---|---|---|
| 10-25 | 建 repo 和项目骨架 | `e328607` |
| 10-28 到 10-29 | 数据下载 `fetch_data.py`（Yahoo + FRED），修 XLC 下载和 concat 问题 | `8274bd5`、`2fefdf6`，PR #1 |
| 11-03 | FOMC 事件表 `build_events.py`（surprise 定义）和 `build_returns.py`（excess return） | `3e27080`，PR #2 |
| 11-04 | 共享的 `config.py`、`data.py`（split、tickers、`load_panel`、metrics） | `0590f81` |
| 11-05 | **GP v0**：pooled + per-ticker，validation 选 parsimony，10 seeds | `2434534` |
| 11-07 | sympy 公式提取（`gp_formulas.csv`）；`compare.py`，加入 zero forecast 参考行和 go/no-go 规则 | `26c1eb8` |
| 11-10 | **linear baseline**（原本是 Person 2 的任务）；重写 Stage 1 计划（`STAGE1_PLAN.md`），合并到 main | `127cb44`、`8cda019` |
| 11-11 | `gp.py --scope`，把 GP v1 拆成两台机器跑；加 robustness flags 任务 | `27c9a75` |
| 11-12 | `linear_interact` baseline；GP 设置统一放进 config；**跑 pooled GP v1** | `cf73d5d` |
| 11-14 | 合并 Abhi 的 per-ticker v1，`compare.py` 合成 `gp_v1`；周六 checkpoint，判定 NO-GO | `9d85443`、`results/checkpoint_2026-11-14.md` |
| 11-15 | **Results freeze**：`src/plots.py` + 6 张 slide 图、overfit 分析、Results/Conclusion slides 文字、writeup 第 4/5 节、本文档；合并到 main 并打 tag `stage1-freeze` | 本次 commit |

接下来（11-16 到 11-22）：把 slides 合成一份 deck、统一格式、组织彩排、做 slides 1、6、8、9、10、12 的展示。

### 5.2 Abhishai Ganta（Person 3）

- 11-03：`ARCHITECTURE.md`、novelty check、最终 `PROJECT_PLAN.md`、Stage 1 社团展示计划（`7c8ed0a`、`6c78e41`）。
- 11-07：**`build_features.py`**（vix、slope_2s10s、beta_1y、mom_20d 等，day −1，无 look-ahead）、24 个测试、GitHub Actions CI、`results/data_report.md`（数据质量检查，发现 2008-10-08 的 SPY 数据问题），PR #3（`ac9ddf5`）。
- 11-11：`gp.py --quick`、`src/interpret.py`；**跑 per-ticker GP v1**，写 `results/gp_interpretation.md`（feature 使用频率、例子公式）（`729224b`、`bfe810e`）。
- 11-11 到 11-13：开始 **Stage 2**（branch `stage2-v1`）：SF Fed 官方 surprise、Ken French 12/49 行业、锁定的 2016+ test set、PySR、多个新 baseline、placebo 和 robustness、forecast combination，写了 `docs/HANDOFF.md` 和 `results/v1/STAGE2_REPORT.md`。
- 未完成：Stage 1 的 robustness flags PR（`TARGET` / `SCHEDULED_ONLY` / `DROP_EVENTS`）没有提交，他转去做 Stage 2 了。
- Slides：Data & Method（周二 11-17 due）、GP v1 结果（周三 11-18 due）。

### 5.3 Snata Mohanty（Person 2）

- Repo 里没有她的 commit；她的部分是 slides 和 Colab 上的初步探索。
- Slides：Motivation + literature 2 页（周二 11-17 due）、Baseline 1 页（周三 11-18 due，用 `results/linear.csv` 和 `fig1`）。
- 原计划里的 linear baseline 代码由 Charles 写了。
- （她的 slides 进度我在 repo 里看不到，需要 Charles 确认。）

### 5.4 Leads（Razeen Khan、Benny Signer）

Review、pre-registration、论文 intro / related work、advisor 联络；`stage1_writeup.md` 第 1 节（Question）和第 6 节（Limitations）归他们。

---

## 6. 还剩什么

### 6.1 这周（Stage 1 展示）

| 日期 | 事情 | 谁 |
|---|---|---|
| 周日 11-15 | 建共享 Google Slides（12 页占位），把 `results/` 和 `results/figures/` 发给团队 | Charles |
| 周一 11-16 到周二 11-17 | Results slides（8、9、10）、Models（6）、Conclusion（12）：文字已在 `docs/slides_results.md`，搬进 deck | Charles |
| 周二 11-17 | Data & Method slides | Abhi |
| 周二 11-17 | Motivation + literature slides | Snata |
| 周三 11-18 | GP v1 slides（Abhi）、Baseline slide（Snata）；Charles 合成整份 deck、统一格式 | 全员 |
| 周四 11-19 | **第一次彩排**（线上、计时 12-15 分钟） | 全员 |
| 周五 11-20 | 修改，可选第二次彩排，发给 leads | 全员 |
| 周末 11-21/22 | **正式展示** | 全员 |

规则：谁的 slides 到期没交，Charles 用现有材料做简短版，不再追。

还没确认：展示时长、是否每个人都讲，需要问 leads。

### 6.2 Stage 1 没做、可以留着的

- Robustness flags（`docs/robustness_flags.md`）：去掉紧急降息、去掉 2008/2020、换 target。Freeze 之后不再跑新模型，放进 Stage 2。
- 2008-10-08 SPY 数据问题的 with/without 对比（data report 说要报告，Stage 1 没跑）。
- `stage1_writeup.md` 第 1 节（leads）、第 3 节（Snata，Baseline）、第 6 节（leads）。

### 6.3 Stage 2（论文，GECCO 2027）

在 branch `stage2-v1`，主要由 Abhi 推进；详细步骤见该分支的 `docs/HANDOFF.md`。现状：数据管线（官方 surprise、Ken French 行业）已完成，dev folds 上的结果和 Stage 1 一致：最好的是一个简单的 CAPM-beta 线性模型（ff12 excess_1d r2_oos +0.018 vs zero +0.011），PySR pooled 又输出了常数。2016+ test set 仍然锁着。剩下：冻结模型列表、解锁一次 test、显著性检验、写论文（abstract 1/18，paper 1/25）。

---

## 7. Repo 地图

| 路径 | 内容 |
|---|---|
| `src/fetch_data.py`、`build_events.py`、`build_returns.py`、`build_features.py` | 数据管线 |
| `src/config.py`、`src/data.py` | 共享设置、split、metrics |
| `src/models/` | `linear.py`、`linear_interact.py`、`gp.py` |
| `src/compare.py`、`src/interpret.py`、`src/plots.py` | 汇总、公式解读、出图 |
| `tests/` | pytest，CI 在每个 PR 上跑 |
| `results/summary.csv` | 所有模型 × 行业 × horizon 的最终数字 |
| `results/go_no_go.md`、`checkpoint_2026-11-14.md` | 决策记录 |
| `results/figures/` | 6 张 slide 图 |
| `docs/STAGE1_PLAN.md` | Stage 1 计划和日程 |
| `docs/slides_results.md` | Charles 负责的 slides 文字 + speaker notes |
| `docs/stage1_writeup.md` | 书面报告（按人分节） |
| `docs/PROJECT_PLAN.md`、`ARCHITECTURE.md` | 两阶段总计划、设计理由 |
