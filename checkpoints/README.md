# 已训练的检查点（纳入版本控制）

`outputs/` 被 `.gitignore` 排除（训练产物大且可复现），**但检查点不能只躺在那里**——
它们是不可复现的算力产物，误删就没了。本目录是"冻结副本"，每个文件都是 `outputs/checkpoints/<run>/` 里对应
检查点的字节副本，SHA256 见下表，可用 `Get-FileHash` 复核。

## 清单

| 文件 | 来源 run | 阶段 / 任务 | 关键成绩（200 回合口径） | SHA256 前 16 位 |
|---|---|---|---|---|
| `p1_rule_best.ckpt` | `p1_rule` | P1 平衡（竖直/±11.5°） | 竖直 100%、±15° 95%、±45° 36.5% | `0DEEFBDBDAF91264` |
| `p2_best.ckpt` | `p2` | P2（±23.5°） | 竖直/±15° 100%、±45° 73%、±180° 20.5% | `3780A8604236A35E` |
| `p3_best.ckpt` | `p3` | P3（±53.5°） | ±45° 87%、±90° 45.5%、±180° 28.5% | `87D2F0B6E32FFFAB` |
| `p4_best.ckpt` | `p4` | P4（±98.5°） | ±45° 99%、±90° 53%、±180° 31% | `4F9A47AC49E70BE1` |
| **`p5_swing_fixed_best.ckpt`** | `p5_swing_fixed` | **P5：当前最强平衡器**（无角度终止） | 竖直/**±15°/±45° = 100/100/99.5%**、±90° 62%、±180° 31.5% | `15176C47CA878B9B` |
| `s1_shape_best.ckpt` | `s1_shape` | S1：hanging 起始 + 能量塑形（第 259 迭代） | ±180° **46%**；(135°,180°] 带 36.7% | `78013ECD79A428C7` |
| **`s2_full_best.ckpt`** | `s2_full` | **S2：全角度 + 能量塑形**（第 164 迭代） | **(135°,180°] 95%**、(90°,135°] 31.7%、(0°,45°] 100% | `09CF8BEC874FB779` |
| `s2_full_last.ckpt` | `s2_full` | S2 末期权重（第 299 迭代，val 0.47–0.72 波动） | 未单独做阶梯评估，保留作对照 | `5F19BBCDFC308370` |

完整 SHA256：

```
0DEEFBDBDAF912643C6F712BAC59766B2FD855A25B3907662629CEBEFC127A21  p1_rule_best.ckpt
3780A8604236A35E6D6560BFC60987EB3C726034492251E452698471AF5CC5C1  p2_best.ckpt
87D2F0B6E32FFFABB799DC946803F8E97C6C2F8868BD5571AFAB9F23CDBEE529  p3_best.ckpt
4F9A47AC49E70BE1E3DD8190452D9AD101E1FA210D1C0D7DD69ADDD464EBBD70  p4_best.ckpt
15176C47CA878B9BB3BA8FBC377FF7ADB8836260865B1182D71DBBD70FA70EED  p5_swing_fixed_best.ckpt
78013ECD79A428C78A521FF15C5944625D76B1BDCE5B258FD9F79440E46D6B16  s1_shape_best.ckpt
09CF8BEC874FB7797E67B9CB76BD8F43C16364A76F51DC8CA8A3DCE6005F60B8  s2_full_best.ckpt
5F19BBCDFC30837088A5834F9B454794CEC6876F5B9A076A1BD5F03FF0919B0E  s2_full_last.ckpt
```

## 怎么用

```powershell
# 评估（注意甩摆类分布必须 --terminate-angle 0）
python scripts\evaluate.py --checkpoint checkpoints\s2_full_best.ckpt `
    --init-mode random --init-angle-limit 180 --terminate-angle 0 --episodes 200

# 能力阶梯（成功率 vs 初始角度）
python scripts\band_report.py --checkpoint checkpoints\s2_full_best.ckpt --terminate 0 --episodes 60

# 继续训练（warm start）
python scripts\train.py --init-from checkpoints\s2_full_best.ckpt --init-mode random `
    --init-angle-limit 180 --terminate-angle 0 --shaping energy --run-name s3 --no-render
```

检查点自带 `env_config`（物理参数、任务配置、是否启用塑形），所以评估会自动复现它训练时的世界；
命令行参数只覆盖你显式给出的项。

## 没有纳入的

`outputs/checkpoints/` 里的其余 run（`p4_oldmetric`、`p5_ent001/003`、`p5_fixcheck/fix2check`、
`p5_swing`、`p6_180_nt`、`s1_hanging`、`cont_p1`、`ppo_cart_upright`、`bc_warmstart`）是
**中间尝试或已否决实验**，留在 `outputs/` 而不外发。其中 `bc_warmstart`（模仿预热，已判定为净负面）
的结论记在 `scripts/imitation_warmstart.py` 的模块 docstring 里，不需要权重也能复现判读。

同样**刻意排除**的还有两次负面实验的权重（结论见 `RESULTS.md` 第 5.5 节）：

| run | 为什么不收 | 后果 |
|---|---|---|
| `s3_band` | 60–90° 窄带集中训练 255 迭代，成功率始终 0 | 其它带未退化，但目标带也没涨 |
| `s4_a1_slow` / `s4_a2_wide` | 速率课程同样 0，且**毁掉了已有能力** | (135°,180°] 100% → **0%**，(90°,135°] 31.7% → 0% |

把它们放进版本控制会把"更差的权重"当成里程碑；要复现判读用 `RESULTS.md` 里的配方即可。
`s4_a1_slow/best.ckpt` 仍在 `outputs/checkpoints/` 里，未删除，只是不外发。
