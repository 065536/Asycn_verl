# Memory Index

> 持久化位置：`/data/250010176/codes/verl/memory/`（系统 memory 目录会被清理，以此处为准）

- [user_profile.md](user_profile.md) — 用户背景、工作方式、研究哲学
- [project_theory.md](project_theory.md) — 当前理论框架：improvement bound → r_t → α*(t) = r_t/L；更新至 2026-04-20
- [algorithm_design.md](algorithm_design.md) — entadapt_initial 的缺陷 + 新框架 α_t = c_t·r̂_t；4.21 诊断：p_min 破坏 scale-invariance → 已删除；4.23 理论诊断 r̂_t ratio estimator instability + Phase 2 三角对照最终设计（M/A/B，均值对齐 2.97e-6）
- [feedback_style.md](feedback_style.md) — 协作偏好：严格推导、不逆向工程、超参要透明；4.22 新增：先验证日志再看结果；先看动力学差异再看最终分数；每次实验只测一个自由度
- [engineering_impl.md](engineering_impl.md) — 工程实现：A1/A2 split + 低频校准步 C；p_min guard 已删除（4.21）；phi_t 退化为零 bug 已修复（4.22 第二轮）；sign-gate 模式已实现 + Bug 16 修复（4.23）
- [bug1_fix_design.md](bug1_fix_design.md) — r-side 状态机完整设计 + 实现 + 代码审查 4 个 bug 及修复（2026-04-20 全部完成）
