# Memory Index

> 持久化位置：`/data/250010176/codes/verl/memory/`（系统 memory 目录会被清理，以此处为准）

- [user_profile.md](user_profile.md) — 用户背景、工作方式、研究哲学
- [project_theory.md](project_theory.md) — **6.12 方向转变**：从 signal fraction LR 转向 RL-aware optimizer 设计；原 r_t 理论框架保留为历史参考
- [algorithm_design.md](algorithm_design.md) — **6.12 核心：pos/neg gradient decomposition + Adam preconditioning diagnosis**；|A⁻|大 ≠ negative update 主导的严格论证；P0/P1/P2 路线图
- [feedback_style.md](feedback_style.md) — 协作偏好：严格推导、不逆向工程、超参要透明；5.19 新增：不要只存聚合指标，存原始数组以便离线重算
- [engineering_impl.md](engineering_impl.md) — **6.12 Pos/neg gradient decomposition 实现**；6.10 Exact lm_head gradient norm；6.8b Gradient preconditioning diagnostic；历史：A²Q, SQ-LR, noise decomposition
- [pg_moment_signal_fraction_lr_design.md](pg_moment_signal_fraction_lr_design.md) — 已关闭：PG-Moment 方案和 parameter-space momentum 方案
- [bug1_fix_design.md](bug1_fix_design.md) — r-side 状态机完整设计 + 实现 + 代码审查 4 个 bug 及修复（2026-04-20 全部完成）
