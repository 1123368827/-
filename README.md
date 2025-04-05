# 通信运营商客户流失预测系统 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3.0-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8-orange)

> 基于机器学习和深度学习的客户流失预测解决方案，助力企业实现精准用户留存

## 🌟 项目亮点
- ​**全流程解决方案**：完整覆盖数据清洗、特征工程、模型构建与部署全流程
- ​**多算法融合**：集成随机森林、决策树、逻辑回归和深度神经网络四大算法
- ​**高精度模型**：最佳模型准确率达**79.05%**，精准识别潜在流失用户
- ​**实战导向**：针对运营商真实业务场景设计，可直接应用于生产环境

## 📊 项目流程图
```mermaid
graph TD
    A[原始数据] --> B[数据清洗]
    B --> C[特征工程]
    C --> D[样本平衡]
    D --> E[模型训练]
    E --> F[模型评估]
    F --> G[策略建议]



sequenceDiagram
    participant 用户数据
    participant 特征工厂
    participant 动态评估模块
    participant 行为分析模块
    participant 模型引擎
    participant 预警系统
    participant CRM系统
    
    用户数据->>特征工厂: 原始数据输入
    特征工厂->>动态评估模块: 生成合约有效性指数
    特征工厂->>行为分析模块: 计算通信质量评分
    动态评估模块->>模型引擎: 结构化特征数据
    模型引擎->>预警系统: 输出流失概率(0-1)
    预警系统->>CRM系统: 推送高风险用户名单
