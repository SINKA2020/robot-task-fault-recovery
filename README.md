# Robot Task Fault Recovery

**Scene-Graph-Based Fault Detection and Recovery for Robotic Manipulation**

基于场景图的机器人操作任务故障检测、诊断与闭环恢复研究代码。

本仓库发布研究系统的核心实现，供阅读、研究和进一步开发。真实场景实验已完成；当前公开内容仅为源码，不包含完整的运行环境和实验复现资源。

## 主要功能

- **视觉感知与场景表示**：处理视觉检测结果，构建对象、状态及空间关系的场景图。
- **任务与预期状态建模**：加载任务计划，生成执行阶段对应的预期状态。
- **执行验证与故障诊断**：比较观测状态与预期状态，识别偏差并进行故障诊断。
- **恢复规划与执行**：根据故障及任务上下文执行重试、修复、清场和恢复流程。
- **机器人与夹爪接口**：封装运动原语、夹爪状态及执行交互。

## 源码结构

```text
src/
└── scene_graph_system/
    ├── __init__.py        # Python 包初始化
    ├── perception/        # 腕部与全局相机检测、分割
    ├── scene_graph/       # 场景图、关系计算与快照
    ├── planning/          # 任务计划、预期状态与闭环调度
    ├── validation/        # 状态比较、验证事务与监测
    ├── diagnosis/         # 故障事件与诊断逻辑
    ├── recovery/          # 恢复管理、修复与清场
    ├── robot/             # 运动原语、执行器与夹爪协议
    ├── configuration/     # 配置加载、编译、校验及服务端逻辑
    ├── tools/             # 标定、就绪检查与诊断工具实现
    └── resources.py       # 包资源与运行数据定位
```

内部 Python 包名沿用 `scene_graph_system`。模块之间使用完整包路径导入。

## 阅读建议

可以按以下顺序了解系统：

1. [场景图](src/scene_graph_system/scene_graph/)：了解对象、状态和关系的表达方式。
2. [任务规划](src/scene_graph_system/planning/)与[执行验证](src/scene_graph_system/validation/)：了解预期状态生成和偏差检测。
3. [故障诊断](src/scene_graph_system/diagnosis/)与[恢复](src/scene_graph_system/recovery/)：了解诊断和恢复流程。
4. [机器人接口](src/scene_graph_system/robot/)：了解算法与硬件执行的连接方式。

## 依赖与发布范围

本代码基于 Python 3 和 ROS 1/catkin 开发。相关 Python 第三方依赖列于 [requirements.txt](requirements.txt)，但仅安装这些依赖不能运行完整系统。

部分模块还需要以下外部依赖：

- ROS 消息与通信包：`rospy`、`std_msgs`、`sensor_msgs`、`geometry_msgs`、`cv_bridge`、`actionlib`、`move_base_msgs`。
- 机器人工作空间包：`rm_msgs`、`blockkit`。
- 自定义消息包：由完整工程的消息定义生成的 `scene_graph_system.msg`。
- 硬件 SDK：RealMan Python SDK（`Robotic_Arm.rm_robot_interface`）及 RealSense 运行环境。

当前发布不包含顶层 `scripts/`、`config/`、`launch/`、`msg/`、`models/`、`web/`、`tests/`，以及 ROS 构建配置。这意味着启动入口、任务与规则配置、消息定义、模型权重、网页资源和完整测试环境需另外配套。

`src/` 中保留了系统集成代码，并非所有模块都能独立导入或运行；例如资源定位和消息导入需要完整工程支持。当前版本适合代码阅读与研究，不提供一键运行或完整实验复现入口。

依赖版本应与所使用的 ROS、Python 和硬件 SDK 环境匹配；本发布未提供经完整验证的依赖版本锁定文件。

## 实验情况

项目已完成真实场景实验。当前源码发布未附实验数据、结果图表或复现资源，因此不在此补充未经提供的实验指标。

## 许可证

项目所有者尚未指定正式开源许可证。[LICENSE](LICENSE) 当前仅记录授权状态，不构成开源许可。正式许可证确定后将更新此文件。
