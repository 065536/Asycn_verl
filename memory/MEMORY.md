# Memory Index

> 持久化位置：`/data/250010176/codes/verl/memory/`（系统 memory 目录会被清理，以此处为准）

- [user_profile.md](user_profile.md) — 用户背景、工作方式、研究哲学
- [project_theory.md](project_theory.md) — 当前理论框架：improvement bound → r_t → α*(t) = r_t/L；更新至 2026-04-20
- [algorithm_design.md](algorithm_design.md) — entadapt_initial 的缺陷 + 新框架 α_t = c_t·r̂_t；4.21 诊断：p_min 破坏 scale-invariance → 已删除；4.23 理论诊断 r̂_t ratio estimator instability + Phase 2 三角对照最终设计；4.27 C310/S 排除 mean-alpha 与 jitter confound；4.28 D3 强 open-loop stage schedule 恢复大部分收益但仍低于 B，支持 support-matched schedule 主轴；4.30 重新定位 split alignment 为 noisy but informative low-frequency signal；5.05 low-frequency ablations 结果接近，noise 问题尚未解决，已启动 alpharlim0.05 / slowema_ret0.95 / windowr_w10 的 seed0/seed1 多 seed follow-up；5.06 读数显示 W10 是当前最 promising 的低频聚合方向，但 seed42 old late drop 大，需重跑 W10 seed42 和补 seed2；B-current 已有历史三 seed，不应说缺 B seeds；r_hat×pg_loss 诊断显示 early/mid opportunity 信号，late 不成立，建议 base schedule + small residual；已启动 current B 与 matched constant 的新日志诊断 run
- [feedback_style.md](feedback_style.md) — 协作偏好：严格推导、不逆向工程、超参要透明；4.22 新增：先验证日志再看结果；先看动力学差异再看最终分数；每次实验只测一个自由度
- [engineering_impl.md](engineering_impl.md) — 工程实现：A1/A2 split + 低频校准步 C；p_min guard 已删除（4.21）；phi_t 退化为零 bug 已修复（4.22 第二轮）；sign-gate 模式已实现 + Bug 16 修复（4.23）；alpha replay controls、D3 piecewise stage-matched decay；4.28 D3 replay sanity-check 通过；新增 7B sync 16/32GPU c-fixed sweep 脚本；7B 首次启动的 debug quantile crash 与 Hydra struct override 已修复；file logger 本地保存 resolved config；4.30 已实现 slow EMA、alpha rate limit、3A windowed continuous-r W5/W10；5.05 主 1.5B c-fixed 脚本支持 SEED env，新增 6 个 seed0/seed1 controller wrapper；新增 PPO ratio tail / clip high-low / reward-advantage dispersion diagnostics，并启动 current B / matched constant 诊断 run；async new-engine r-side port for c-fixed sweep（C-side 未迁移）
- [bug1_fix_design.md](bug1_fix_design.md) — r-side 状态机完整设计 + 实现 + 代码审查 4 个 bug 及修复（2026-04-20 全部完成）
