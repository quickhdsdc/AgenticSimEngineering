# 委托建模任务：KUKA 合成数据辨识

用户已授权 Codex 根据上下文作为需求代理补充信息，并使用人造数据走通建模流程。此授权无需再次询问。最终再制造应用与真实部署设备尚未确定。

你使用既有 A4S Harness 完成“资料/数据 → 模型生成 → 参数辨识 → 留出实验验证”。先通过一次 ask_user 明确本次探索的具体辨识目标和验收标准，让需求代理给出带来源标记的补充；然后使用固定 CLI 完成各阶段。

参考模板是 examples/kuka-calibration.json：固定版本的 KUKA iiwa 14，暂以 joint2 黏性阻尼与 armature 为两个未知参数。模板初值、噪声和误差阈值是工程探索假设，不能标成厂家参数或用户逐项确认值。

对话与执行规则：

- 明确区分 delegated_operator 需求补充、synthetic_fixture 数据和 reference model 资产。没有真实测量不要标成 measurement。
- 读模板后根据需求代理回答形成计划，用 generate/model/fit/validate 的精确命令依次执行，不重跑已经保存的阶段。
- 生成器创建两组训练实验、一组不同频率/随机种子的留出实验；真值保留在私有 fixture，不能读取，也不能把留出样本用于调参。
- 模型生成写出实际 seed-model.xml；拟合输出实际 identified-model.xml，不能只写建模建议。
- 读取 fit.json、validation.json 的参数、单位、来源、哈希和误差，将解释保存为会话 artifacts/calibration-summary.md。
- 报告仅对 synthetic_holdout_only 有效，physical_validation 保持 not_established。独立实验分割不等于独立真实设备验证。
- 当前不连接设备，不执行 PLC/MES/AAS 写入，不下载另一台机器人的轨迹来冒充此设备数据。
- 其他未知目标由 ask_user 向需求代理询问；用户已经授权的上下文不重复询问。
