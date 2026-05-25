# Memory Index

> 持久化位置：`/data/250010176/codes/verl/memory/`（系统 memory 目录会被清理，以此处为准）

- [user_profile.md](user_profile.md) — 用户背景、工作方式、研究哲学
- [project_theory.md](project_theory.md) — 当前理论框架：improvement bound → r_t → α*(t) = r_t/L；更新至 2026-04-20
- [algorithm_design.md](algorithm_design.md) — **5.20 SQ-LR 完成**：离线验证 + sensitivity 分析（reward_std best, bernoulli_var fails）+ 分布式一致性修复 + Group C 待启动；seed1 是 info-only 薄弱点，可能需要 Group D concentration gate
- [feedback_style.md](feedback_style.md) — 协作偏好：严格推导、不逆向工程、超参要透明；5.19 新增：不要只存聚合指标，存原始数组以便离线重算
- [engineering_impl.md](engineering_impl.md) — **5.20 SQ-LR 工程实现**：SignalQualityLRScheduler + all_reduce 分布式一致性 + uid prompt grouping + 16 metrics/step（含 4 个替代指标）+ sensitivity 脚本
- [pg_moment_signal_fraction_lr_design.md](pg_moment_signal_fraction_lr_design.md) — 已暂停：PG-Moment 方案和 parameter-space momentum 方案
- [bug1_fix_design.md](bug1_fix_design.md) — r-side 状态机完整设计 + 实现 + 代码审查 4 个 bug 及修复（2026-04-20 全部完成）
