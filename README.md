# 强化学习控制单阶倒立摆 (PPO + PyTorch Lightning)

用强化学习（PPO）控制**单阶倒立摆**：小车在水平导轨上运动，杆子铰接在小车上，
智能体只能施加水平推力，目标是把杆子**摆起**并**稳定倒立**在竖直向上的不稳定平衡点。

```
                    ● 杆端 (m, l)
                   /
                  /   theta = 0 表示竖直向上
                 /
        ┌───────┐
        │ 小车  │ ──────► x     智能体输出力 F ∈ [−50, 50] N
        └───────┘
   ══════════════════════════  导轨  |x| ≤ 2.4 m
```

* 物理环境：自己实现的 Gymnasium 兼容环境，RK4 积分，无第三方物理引擎
* 算法：PPO（从零实现，裁剪代理目标 + GAE + 熵正则）
* 训练框架：**PyTorch Lightning**（`LightningModule` + `LightningDataModule`，手动优化模式）
* 输出：TensorBoard 日志、JSONL 指标、checkpoint、**实时窗口**、GIF/MP4 动画

---

## 1. 快速开始

```powershell
# 环境（本仓库已验证：Python 3.11 + torch 2.14 CPU + pytorch-lightning 2.6.6）
conda create -n pendrl python=3.11 -y
conda activate pendrl
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 纯 CPU 版，体积小
pip install -r requirements.txt

# 0) 物理自检（21 项，强烈建议先跑，约 40 秒）
python tests/test_sanity.py

# 1) 训练"平衡"任务（从接近竖直的状态出发，保持 10 秒）
#    默认会开一个实时窗口，边训练边看倒立摆
python scripts/train.py --model cart --init-mode upright --run-name ppo_upright

# 2) 实时观看训练好的策略（真速度播放，直到你关窗）
python scripts/evaluate.py --checkpoint outputs/checkpoints/ppo_upright/best.ckpt --max-seconds 0

# 3) 和经典控制器对比（零输入 / 随机 / PD / 能量起摆 / PPO）
python scripts/compare.py --checkpoint outputs/checkpoints/ppo_upright/best.ckpt

# 4) 画训练曲线 / 生成汇总报告
python scripts/plot_training.py
python scripts/report.py
```

TensorBoard：`tensorboard --logdir outputs/logs --port 6006`

> **实时窗口默认开启**：训练和评估都会弹出一个 tkinter 窗口实时显示倒立摆
> （跟随 0 号并行环境，跳帧刷新，关掉窗口不会中断训练）。
> 想跑得更快或没有显示器时加 `--no-render`；评估时想导出 GIF 用 `--video`。

---

## 2. 初始状态决定任务

"倒立摆从上面还是下面开始"由 `--init-mode` 决定，这正是任务的定义方式：

| `--init-mode` | 初始姿态 | 任务 |
|---|---|---|
| `upright`（**默认**） | 接近竖直，θ ∈ ±0.05 rad（±3°），静止释放 | **平衡**：从上面开始，稳住不放倒 |
| `hanging` | 垂在下方，θ ≈ 180° ± 0.2 rad | **起摆**：从下面甩上去再接住 |
| `random` | 默认整圈均匀 θ ∈ [−π, π]；配合 `--init-angle-limit` 可收窄范围 | **鲁棒控制**：任意初始状态都能救回 |

实测这三种初始分布需要的策略**确实不一样**，不能互相替代（同一模型、20 回合）：

| 训练时的初始分布 | 竖直起始 | ±15° 起始 | ±45° 起始 |
|---|---|---|---|
| `upright`（阈值 11.5°） | **撑满 10 s，θ RMS 0.4°** | 0/20 倒下，θ RMS 2.4° | 5/20 倒下 |
| 直接上 `random`（整圈） | 可以，但很弱 | — | 成功率约 20% |

只学过"接近竖直"的策略，遇到"杆子已经倒下"的状态不会救。**难度必须逐级加上去**，
这直接引出下一节的课程设计。

---

## 3. 课程设计：为什么它是一套连续、统一的规则

### 3.1 问题：一条"倒下就结束"的规则和起摆是冲突的

平衡任务的标准做法是"杆子超过某个角度就结束回合"——原始 CartPole 用的就是 ±12°
（[cartpole.py](https://zoo.cs.yale.edu/classes/cs470/materials/hws/aima/gym/gym/envs/classic_control/cartpole.py)：
`done = x < -2.4 or x > 2.4 or theta < -12° or theta > 12°`，奖励是每步 +1，
**总回报 = 存活步数**，判定"解决" = 连续 100 回合平均回报 ≥ 195）。

但起摆要求杆子从 180° 转到 0°，**途中必然经过 90°**，任何固定的角度阈值都会让起摆回合在开局第一步结束。
而 Gymnasium 的起摆基准 Pendulum 则完全不终止：它 "never terminates on its own -- it relies entirely on
the TimeLimit wrapper"，初始状态取 θ ~ U(−π, π)
（[PendulumEnv](https://leeroopedia.com/index.php/Implementation:Farama_Foundation_Gymnasium_PendulumEnv)）。

两个官方基准各用一套规则，因为它们各自只做一个任务。**要在同一个项目里做完整课程，就必须让规则本身可扩展。**

### 3.2 方案：固定规则的形式，只让参数单调放宽

$$
\text{回合结束} \iff |\theta| > \theta_{\text{fail}}(k) \;\lor\; |x| > 2.4\,\text{m} \;\lor\; \text{数值发散},
\qquad \theta_{\text{fail}}(k) = \theta_{\text{init}}(k) + m
$$

$k$ 是阶段序号，$m$ 是固定余量，$\theta_{\text{init}}(k)$ 是**该阶段初始角度的范围半径**：

| 阶段 | 初始分布 θ₀ ~ U(−θ_init, +θ_init) | 阈值 θ_fail = θ_init + m | 命令行 |
|---|---|---|---|
| P1 平衡 | ±3° | **±11.5°**（≈ CartPole 的 ±12°） | `--init-mode upright --terminate-angle 11.5` |
| P2 小角度 | ±15° | ±23.5° | `--init-mode random --init-angle-limit 15 --terminate-angle 23.5` |
| P3 中等 | ±45° | ±53.5° | `--init-angle-limit 45 --terminate-angle 53.5` |
| P4 大角度 | ±90° | ±98.5° | `--init-angle-limit 90 --terminate-angle 98.5` |
| P5 近全圆 | ±135° | ±143.5° | `--init-angle-limit 135 --terminate-angle 143.5` |
| P6 全圆 | ±180° | **无角度条件** | `--init-angle-limit 180 --terminate-angle 0` |

两个关键性质：

1. **形式全程统一**。所谓"前几个阶段不倒就得分"和"后面允许起摆"**不是两条规则**，
   而是同一条规则在不同参数下的表现——和 CartPole 的 ±12° 完全同构（CartPole 只是把参数定死了）。
2. **终点自然退化到官方约定**。当 θ_init = 180° 时阈值也到 180°，而缠绕后 |θ| ≤ π，
   于是 {|θ| > 180°} = ∅：**角度条件自动消失**，只剩导轨条件，正好等于 Pendulum 的设定。
   P6 不需要任何特例代码。

### 3.3 为什么它是连续的：四个可独立验证的命题

**(A) 目标函数逐位相同 —— 数值连续。**
阶段切换只改"初始分布"和"阈值"，这两个量都**不出现在奖励里**，各项权重也全程固定：

$$
r(s,a) = \underbrace{\cos^2\theta + c_{\text{off}}}_{\text{存活项}} - w_\theta\theta^2 - w_x x^2 - w_v\dot x^2 - w_\omega\dot\theta^2 - w_u (u/u_{\max})^2
$$

所以同一状态在任何阶段算出的奖励**完全相等**。奖励定义"什么算好"，它不变就意味着**"好"的定义不变**；
阶段之间变的只是"哪些局面会出现、哪些局面算结束"。

**(B) 约束集单调收缩 —— 单调连续。**
记失败集 F_k = { |θ| > θ_init(k) + m }，则

$$
\mathcal{F}_1 \supseteq \mathcal{F}_2 \supseteq \cdots \supseteq \mathcal{F}_6 = \varnothing
\quad\Longrightarrow\quad
\mathcal{S}_1 \subseteq \mathcal{S}_2 \subseteq \cdots \subseteq \mathcal{S}_6 = \mathcal{S}
$$

由此得到课程学习最需要的性质：**在阶段 k 学到的任何策略，到阶段 k+1 的回报只会变好或不变，绝不会变差**
（奖励相同 ⇒ 已有行为的得分不变；失败集收缩 ⇒ 轨迹只可能更长）。
旧本事不会因规则变化而被惩罚，这就是"换规则不会打崩训练"的严格理由。

反过来，**非单调**（先宽后严）会造成"突然死亡"：过去能拿回报的行为被瞬间判罚，价值函数与策略同时受冲击。
**单调性是安全方向的分界线。**

**(C) 参数连续 ⇒ 极限无跳跃。**
阈值 θ_fail = θ_init + m 随参数连续变化，θ_init → π 时 θ_fail → π、F → ∅，中间没有突变点：
每推一点只多放进来一圈状态。P5→P6 不是换规则，而是**沿同一条曲线走到端点**。

**(D) 奖励在终止边界处也连续。**
终止时**不加额外惩罚**（代码中 `reward -= 0.0` 就是留出的位置），所以
lim_{θ→θ_fail⁻} r 与到达该状态时的奖励一致，V 只是**停止累加**（V(s_term) = 0）
而非被跳变的惩罚打断——价值函数在阈值两侧不会断裂。

### 3.4 实测验证

四个命题都做了实验：以 P1 权重切到 P2 规则。

**(A) 奖励同一性**：300 个随机状态上，两阶段奖励的最大差值 = **0.000e+00**（恰好为 0，不是"很小"）。

**(B) 价值函数迁移**：

| 分布 | E[V] | sd[V] |
|---|---|---|
| P1（竖直起始，阈值 11.5°） | 74.78 | 1.50 |
| P2（±45° 起始，阈值 60°） | 46.18 | 35.49 |

E[V] 保持同量级、没有爆炸式跳变；sd[V] 从 1.5 涨到 35.5 是**应该的**——P2 的状态价值本来就该有分布。
若奖励不连续，E[V] 会跳到完全不同的量级。

**(C) 实际训练**（从 P1 权重直接切到 P2 规则，只跑 40 个迭代）：

| 评估分布 | 切换前 | 40 迭代后 |
|---|---|---|
| P2（±45° 起始） | 回报 139.4，成功 **25%** | 回报 378.3，成功 **80%** |
| P1（竖直起始） | 回报 496.9，成功 **100%** | 回报 492.5，成功 **100%**（**未退化**） |

两处细节：**P1 权重在 P2 上开箱就有 25%**，说明能力是**迁移**过去的而非从零重学（命题 B）；
**P1 能力没有退化**，说明单调放宽约束不会造成灾难性遗忘。

### 3.5 什么会破坏连续性（本项目刻意避开的坑）

| 做法 | 后果 |
|---|---|
| 阶段间改奖励权重 | 命题 A 失效，V 的目标含义变了，warm-start 变得有害 |
| 阈值先宽后严（非单调） | 命题 B 失效，出现"突然死亡" |
| 终止时给阶段相关的额外惩罚 | 边界出现奖励悬崖，V 在阈值两侧断裂 |
| 阶段间改控制频率 / 作动器限幅 | 动力学或奖励尺度改变，迁移被破坏 |
| 换环境实现但物理不完全一致 | 仓库里有**批量物理与标量物理逐位一致**的测试（差值 0.00e+00）专门防这条 |

另外两个**不影响连续性、但影响训练稳定性**的实务点：

* **观测归一化的运行统计**每阶段需要几十个迭代适应新分布，头几步 V 会有数值暂态（是"暂态"不是"不连续"）。
* **学习率 / 熵系数的调度**要按阶段重置进度，否则新阶段会在退化后的学习率上起步。

### 3.6 一句话总结

> **奖励（"什么算好"）逐位不变，只有约束集沿一条单调曲线逐步收缩；所有难度增长都通过"初始状态分布"这一个通道发生，而不是通过修改目标。因此各阶段是"同一个问题的不同难度"，而不是"不同的问题"。**
>
> 换句话说：**P1 到 P6 是同一条规则在参数空间里走的一条单调路径，CartPole（±12°）与 Pendulum（不终止）只是这条路径的两个端点。**

---

## 4. 实时窗口

`pendulum_rl/live_view.py` 用 **tkinter** 实现（CPython 自带，无需 pygame/OpenCV）。
绘制开销约 1–2 ms/帧，而且会**主动跳帧**，所以窗口永远不会拖慢训练；关掉窗口后训练照常继续。
PPO 更新与验证阶段也会持续刷新并显示当前阶段（`PPO update (10 epochs × 16 minibatches)` /
`evaluating (x/500 steps)`），所以不会出现"窗口假死卡在 iteration 0"。

窗口中显示：

* 倒立摆本体（小车、杆、竖直参考虚线、导轨 0.5 m 刻度、±x_limit 红标），**镜头跟随小车**，
  漂移量由刻度体现（否则稳定但漂移的小车会走出画面）
* 顶部：PPO 迭代数、累计环境步数、回合号、rollout 进度条
* 中间：`BALANCED` 提示（|θ| 进入成功阈值时亮起）
* 底部：θ / x / u（含占限幅百分比）、单步回报、**本环境近 100 步 |θ| 滑动平均**、
  全部并行环境的瞬时均值、成功率、控制量条形图
* 评估模式另有 `holding for XX.X s`（不限时，直到杆子真的倒下）

无显示器时自动降级为每 1 秒写一张 PNG 到 `outputs/live/<run>.png`，再不行就静默关闭，
绝不因为可视化而中断训练。

评估时的实时播放按**真实时间**节流（`control_dt` = 20 ms/步），看到的速度就是真实的物理速度：

```powershell
# 一直播放直到你关窗（--max-seconds 0 = 不限时）
python scripts/evaluate.py --checkpoint <ckpt> --init-mode upright --max-seconds 0

# 想看它在更难的开局下表现如何（比训练范围更难，会看到救回与失败）
python scripts/evaluate.py --checkpoint <ckpt> --init-mode random --init-angle-limit 45 --terminate-angle 60
```

---

## 5. 任务与环境设计

| 项目 | 设定 | 说明 |
|---|---|---|
| 状态 | `[x/x_limit, ẋ/10, cosθ, sinθ, θ̇/8, θ/π]` | 共 6 维，已缩放到 O(1)；用 `cos/sin` 表示角度避免 ±π 跳变 |
| 动作 | 水平力 `F ∈ [−50, 50] N` | 连续控制。50 N 是刻意留的裕度：15 N 时稳定控制器长期把限幅打满（实测约 97% 饱和），学习极难 |
| 物理步长 | 2 ms（RK4） | 每 10 个子步产生一次决策 → 50 Hz |
| 回合长度 | 500 步 = 10 s（`max_episode_steps` 可设 `None` = 无时限） | |
| 奖励 | `cos²θ + 0 − 3θ² − 0.1x² − 0.01ẋ² − 0.1θ̇² − 0.001(ũ)²` | θ 为**缠绕后**的角度；竖直静止时恰为 **+1.0/步**（即 CartPole 的"存活 = 得分"），500 步满分 500 |
| 结束条件 | `\|θ\| > θ_fail`（**终止**）或 `\|x\| > 2.4 m` / 数值发散（**截断**） | 阈值随课程单调放宽，见第 3 节 |
| 成功判据 | `\|θ\| < 0.12 rad` 且 `\|x\| < 1.5 m` 连续保持 100 步（2 s） | 仅用于评估，不参与训练 |

关于 `cos²θ`：它**不能**换成 `cosθ`。用 `cosθ` 时"杆子一直转圈"的平均收益和"稳稳立住"几乎相同
（转圈时 cosθ 均值为 0），梯度会去选更容易的转圈策略；换成 `cos²θ − 3θ²` 后只有竖直附近才是正收益
（30° 处已经变负），"立住"才严格优于"转圈"。

两种被控对象（`--model`）：

* `cart`（默认）：小车 + 摆，**这就是通常说的倒立摆**
* `pivot`：转轴固定、直接加力矩（等价于 `gymnasium` 的 `Pendulum-v1`），学起来更简单，用于交叉验证

---

## 6. 算法：从零实现的 PPO

`pendulum_rl/agents/ppo.py`，约 300 行，无 RL 框架依赖：

* 高斯策略 π(a|s)，`log σ` 为**与状态无关**的可学习参数（低维控制的常规做法）
* 裁剪代理目标（`clip_range=0.2`）+ 裁剪价值损失 + 熵奖励
* GAE(λ=0.95)、γ=0.99，优势归一化
* 观测**运行均值/方差归一化**（随 checkpoint 保存，推理时冻结）
* `target_kl=0.1`：KL 超过阈值提前结束该轮更新
* 熵系数从 1e-3 线性退火到 1e-4；正交初始化、actor 输出层 gain=0.01
* 策略初始 `log σ = −3`（σ ≈ 0.05，满量程的 5%）——**这是本项目最关键的调参**：
  倒立摆的最优控制量极小（实测均值 0.05 N / 50 N），若按常见默认 σ ≈ 0.6 起步，
  探索噪声比最优动作大一个数量级，策略永远学不动

**为什么用 Lightning 这样组织**（`pendulum_rl/lightning_module.py`）：

* PPO 是 on-policy 的，"数据集"就是**最新策略采出的 rollout**，所以用 `LightningDataModule`
  承载"一次迭代 = `num_envs × rollout_steps` 条交互"，配合
  `reload_dataloaders_every_n_epochs=1` 每个 epoch 重新采集。
* Lightning 2.x 在 `automatic_optimization = False` 时会给 `training_step` 套 `torch.no_grad()`，
  因此更新段用 `torch.enable_grad()` 包住，并由 `PPOAgent` 自己做多次 minibatch 更新。
* `on_save_checkpoint` 把 agent 权重、观测归一化统计、**生成该模型的配置**一起存进 checkpoint，
  推理时 `load_policy()` 一键重建，不需要手动对齐超参。
* 指标同时写 TensorBoard 和 `outputs/logs/<run>/metrics.jsonl`（不装 TensorBoard 也能读）。

**性能**：环境是批量的（`pendulum_rl/batched_env.py`），16 个环境同时用 NumPy 积分，
瓶颈在策略网络而不在物理。批量实现与标量实现有**逐位一致**的单元测试保护。

---

## 7. 目录结构

```
pendulum_rl/
  envs/inverted_pendulum.py   # 标量 Gymnasium 环境（参考实现，含渲染）
  batched_env.py              # 向量化物理/奖励（训练与评估实际使用）
  vector_env.py               # SyncVectorEnv：批量环境 + 回合统计
  agents/ppo.py               # PPO（ActorCritic / RolloutBuffer / PPOAgent）
  agents/classical.py         # LQR（Hamiltonian 特征向量法解 CARE）、PD、能量起摆
  lightning_module.py         # TrainConfig / RolloutDataModule / PPOLightningModule / 检查点回调
  live_view.py                # tkinter 实时窗口
  rendering.py                # matplotlib 渲染 + GIF/MP4 + 轨迹图
  utils.py                    # 运行均值方差、指标写入、路径
scripts/
  train.py                    # 训练入口（课程、实时窗口、warm-start）
  evaluate.py                 # 评估 + 实时窗口 + 视频/轨迹图
  compare.py                  # PPO vs 经典控制器
  plot_training.py            # 训练曲线
  report.py                   # 汇总报告
tests/test_sanity.py          # 21 项物理/算法自检
```

---

## 8. 物理正确性怎么保证的

倒立摆的符号约定（θ 正方向、推力对杆的反作用、能量与动量）极易写错，
所以本项目的流程是**先用经典控制验证物理，再上强化学习**。

`python tests/test_sanity.py` 检查 21 项：

1. **重力使竖直平衡失稳**：θ = 0.05 rad 释放后 θ 单调增大（线性化特征值 +5.24 /s）
2. **无控制必倒**：整个回合内峰值 |θ| > 1 rad
3. **能量守恒**：零输入 5 s，总机械能漂移 < 1e-6 J（实测 **1.1e-12 J**）
4. **动量守恒**（小车模型）：零推力时水平动量漂移 < 1e-6 kg·m/s（实测 **1.3e-12**）
5. **非最小相位**：向右推小车，竖直杆先向**左**倾（倒立摆的经典特征）
6. **PD 控制器能稳定 10 秒**（20/20 个种子）
7. **能量起摆 + LQR 能摆起并稳定**（转轴模型）
8. **批量物理与标量物理逐位一致**（cart/pivot × 三种初始状态，差值 = 0.00e+00）
9. 终止规则、PPO 更新、渲染等冒烟测试

前 4 项同时成立基本排除了动力学写错的可能（推导过程中确实抓出了两处符号错误）。

### 能量起摆的符号（踩坑记录）

摆起控制器用的是 Åström & Furuta 能量整形法，两个恒等式经过**数值实测**确定：

* 转轴模型：`dE/dt = +τ·θ̇` ⟹ 抽能律 `τ = −k·(E−E*)·θ̇`
* 小车模型：`dE_pend/dt = −m·l·a·θ̇·cosθ` ⟹ 抽能律 `a = +k·(E−E*)·θ̇·cosθ`，
  且必须再把加速度换算成力 `F = m_eff(θ)·a`，其中 `m_eff = M + m(1 − ¾cos²θ)`

手推容易错（本项目在这步错了两次），`test_sanity.py` 里的守恒律检查才是最终判据。

**另一处坑**：交接给 LQR 的"捕获域"必须**足够宽**（取 `|θ| < 1.5 rad`、`|θ̇| < 10 rad/s`）。
能量律会把杆子加速到以 ~8 rad/s 冲过顶点，窄窗口几乎永远进不去，杆子就一直空转；
宽窗口下 LQR 同时起到"刹车"作用。

---

## 9. 命令行参数速查

```powershell
python scripts/train.py --help
python scripts/evaluate.py --help
```

训练常用：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model {cart,pivot}` | `cart` | 被控对象 |
| `--init-mode {upright,hanging,random}` | `upright` | 初始状态分布 |
| `--init-angle-limit N` | 无（整圈） | `random` 模式收窄到 ±N**度**，课程主旋钮 |
| `--init-rate-limit N` | 0.5 | 初始角速度上限 (rad/s) |
| `--terminate-angle N` | 0.6° | `\|θ\| > N` **度**即终止；`0` = 不因角度终止 |
| `--warmup-init-mode` / `--curriculum-fraction` | – / 0.4 | 初期用另一种初始模式 |
| `--init-from <ckpt>` | – | warm-start：只迁权重与观测归一化统计，不迁优化器状态 |
| `--max-episode-steps` | 500 | 回合上限 |
| `--max-force` | 50 | 作动器限幅 |
| `--w-th / --w-x / --w-omega / --w-u` | 3.0 / 0.1 / 0.1 / 0.001 | 奖励权重 |
| `--log-std-init` | −3.0 | 策略初始探索噪声（见第 6 节，关键调参） |
| `--render` / `--no-render` | **开** | 实时窗口 |
| `--fast-dev-run` | – | 1 个 epoch 冒烟测试 |

评估常用：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--checkpoint` / `--baseline {random,pd,energy,zero}` | – | 二选一 |
| `--episodes` | 5 | 回合数 |
| `--max-seconds` | 跟随 checkpoint | 单回合模拟时长预算；`0` = 不限时（配合 `--render`） |
| `--max-steps` | – | 直接给步数上限，优先于 `--max-seconds` |
| `--wall-timeout` | 900 | 整体真实时间兜底，防止无界面跑飞 |
| `--terminate-angle N` | 跟随 checkpoint | 覆盖终止阈值（**度**） |
| `--init-angle-limit N` | 跟随 checkpoint | 覆盖初始范围（**度**） |
| `--render` / `--no-render` | **开** | 实时窗口 |
| `--video` / `--no-video` | 开窗时自动跳过 | 导出 GIF/MP4 |

---

## 10. 结果

所有训练产物在 `outputs/`（已在 `.gitignore` 中排除，可复现）：

* `outputs/logs/<run>/metrics.jsonl`、`outputs/logs/<run>/tb/`（TensorBoard）
* `outputs/checkpoints/<run>/{best,last}.ckpt`（`best` 按验证回报刷新，`last` 每迭代覆盖写）
* `outputs/videos/*.gif`、`*_trajectory.png`（评估时生成）

已实测的平衡策略（`upright`，阈值 11.5°）：

| 指标 | 数值 |
|---|---|
| 验证回报 | **496.9 / 500** |
| 竖直起始 20 回合 | **20/20 撑满 10 s**，θ RMS 0.39°，控制量占限幅 **0.07%** |
| 加长回合到 60 s / 300 s | 2994.6/3000、14973.5/15000，θ RMS 0.16° / 0.07°，**零失败** |

**注意**：10 秒只是训练时的回合上限，策略本身能无限期稳定（`--max-seconds 0` 实测如此）。

---

## 11. 说明与已知限制

* **评估口径**：默认自动跟随 checkpoint 记录的训练回合长度（500 步 = 10 s）。
  如果把单回合预算设成远大于训练上限，一局里会包含多次重置尝试，`mean hold`、"never fell"
  这类数字会**明显虚高**。想看真实能力就用默认口径，或显式 `--max-seconds 10`。
* **`--terminate-angle` 与 `--init-angle-limit` 都以"度"为单位**（内部转弧度）。
* **起摆/大角度阶段必须关掉角度终止**（`--terminate-angle 0`），否则开局即结束——见第 3.1 节。
* `pip install pytorch-lightning` 只提供 `pytorch_lightning` 包名，不包含统一的 `lightning`
  命名空间包，所以代码统一 `import pytorch_lightning`。
* 经典能量起摆控制器在**转轴模型**上稳定成功（`test_sanity.py` 硬性断言）；
  在**小车模型**上只能把杆子抽到约 80–120% 临界能量——因为小车模型里推力要先克服等效惯量，
  且导轨长度有限，能量律的加速度假设会被饱和破坏。

---

## 12. GPU / CUDA

**代码层面完全设备无关**，没有任何地方把 device 写死：

| 位置 | 行为 |
|---|---|
| `PPOAgent(device=...)` | `resolve_device()` 把 `auto` / `gpu` 映射为 `cuda`（可用时），否则 `cpu`，也接受显式 `cuda` / `cpu` / `mps` |
| 网络与可学习参数 | `ActorCritic(...).to(self.device)`，`log_std` 同设备 |
| PPO 更新的每个张量 | 全部用 `torch.as_tensor(..., device=self.device)` 显式放在同一设备上 |
| Lightning | `Trainer(accelerator="auto" if --device auto else --device)` |
| checkpoint 读取 | `torch.load(..., map_location=...)`，跨设备可读 |
| 环境 / 滚动缓冲 / 渲染 | 纯 NumPy 与 Python，**始终在 CPU**，不需要也不应该搬到 GPU |

所以：**换到有 N 卡的机器上，只要装对 torch wheel，`--device auto` 就会自动用 GPU，代码不用改。**

### 在有 NVIDIA 显卡的机器上装环境

```powershell
conda create -n pendrl python=3.11 -y
conda activate pendrl

# 关键：装 CUDA 版 torch（不是 +cpu 那个）。CUDA 版本按你的驱动选，
# 见 https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

# 确认装对了：torch.version.cuda 不应该再是 None
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python tests/test_sanity.py
```

### 在本机检查（当前这台是 CPU-only 的 wheel）

```
torch.__version__          = 2.14.0+cpu
torch.version.cuda         = None        <- CPU-only wheel
torch.cuda.is_available()  = False
device_count               = 0
```

代码解析仍然正确（`--device cuda -> cuda`，只是没有卡可用）——**所以本机看到的
`is_available() = False` 是 "torch 装的是 CPU 版"，不是 "程序不支持 CUDA"。**

### checkpoint 跨机器通用

PyTorch 的 `torch.save` 总是把张量存成 CPU 表示，读取时用 `map_location` 再搬到目标设备。
因此：

* 本机（CPU）训出的 `outputs/checkpoints/cont_p1/best.ckpt` **可以直接拿到 GPU 机器上 warm-start**：
  ```powershell
  python scripts/train.py --init-from <path>\cont_p1\best.ckpt --device auto ...
  ```
* 反过来，GPU 机器上训出的 checkpoint 也能在本机 CPU 上评估/续训。

### 但是：这个任务的 GPU 加速比有限，请先量一下

**环境仿真是纯 NumPy，跑在 CPU 上**，而它占了训练时间的大头：16 个环境 × 256 步 ×
10 个物理子步，每个 iteration 要积分约 4 万个 RK4 子步。网络只有 2×256，
前向/反向在这个规模上非常便宜。

所以实际预期是：

* **采样（rollout）阶段完全不受益于 GPU**，它由环境决定；
* **更新阶段（PPO 的 10 epochs × 16 minibatches）会明显变快**；
* 端到端加速比大概在 **1.2–2×** 量级，而不是 10×。

如果换成大网络（`--hidden-sizes 512 512 512`）、或把 `--num-envs` 提到几百，
GPU 的收益才会明显。想量化的话，同一配置各跑几十个 iteration 对比
`perf/rollout_s` 与 `perf/update_s`（两个指标都已写进 `metrics.jsonl`）：

```powershell
python scripts/train.py --run-name gpu_bench --max-epochs 30 --device cuda --no-render
python scripts/train.py --run-name cpu_bench --max-epochs 30 --device cpu  --no-render
python -c "from scripts.show_metrics import *"   # 或直接看两个 metrics.jsonl 的 perf/* 字段
```

### 其它设备的注意事项

* `--device` 取值：`auto`（默认）/ `cpu` / `cuda` / `mps`（Apple）。传 `gpu` 也会被解析为 `cuda`。
* 传了 `--device cuda` 但机器没有可用 GPU 时，Lightning 会直接报错（这是好事：避免悄悄退回 CPU）。
* 实时窗口（tkinter）与 GIF 导出（matplotlib + imageio）都在 CPU 上，不影响设备选择。
* `--device mps` 在本项目未做专门验证；`mps` 上个别算子可能缺实现，如遇报错请退回 `cpu`。

