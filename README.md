# AI 图片识别与开放集平台归因

这是项目的展示与部署仓库，包含两部分：

1. `GitHub Pages` 静态展示页
2. `Render` 可推理 Flask 服务

## 说明

这个版本用于在线展示和技术验证，包含：

1. 网页原型的视觉结构
2. 典型样本切换
3. 四平台候选概率与开放集拒识
4. 文件层、视觉统计与频域信号解释
5. 自动通过、人工复核与高风险核验分流
6. 复核案件创建、人工结论和 CSV 审计导出

## 静态展示版限制

GitHub Pages 不能运行 Flask / Python 后端，因此这个仓库是 **静态展示版**，不是在线实时推理系统。

## 可推理版本

仓库内已经提供 Render 所需文件：

- `app.py`
- `render.yaml`
- `requirements.txt`
- 模型自动还原模块 `model_payload.py`
- 开放集配置 `model_artifacts/platform_open_set_config.json`
- 抗传播模型 `model_artifacts/platform_robust_model_bundle.joblib`
- 盲测摘要 `model_artifacts/platform_open_set_validation.json`

可推理版本会提供：

1. 图片上传
2. `AI / real` 判断
3. 原生导出模型与抗传播模型融合
4. 已采样平台归因与未知平台拒识
5. 文件层、边缘和高频特征解释
6. 三档阈值策略：运营低误伤、均衡复核、高风险售后复核
7. 复核案件 API 与审计导出

## 开放集归因逻辑

1. `AI / real` 模型先判断图片是否进入生成图区。
2. 原生导出模型读取格式、尺寸、EXIF、alpha 和信息键等文件层痕迹。
3. 抗传播模型读取颜色、边缘、噪声与高频能量等视觉统计。
4. 两路概率经温度缩放与加权融合后，还需通过置信度、前两名差值与分布漂移检验。
5. 任一条件不足时输出 `unknown_platform`，候选概率只作为人工核验线索。

## 复核数据

系统默认将复核队列写入 `/tmp/aigc_review_queue.sqlite3`，服务重启后可能丢失。生产部署时应将 `AIGC_REVIEW_DB` 设为持久磁盘路径。队列默认不保存用户原图，仅保存 SHA-256 文件哈希、模型证据、备注和人工结论。

## 验证

```bash
python -m unittest discover -s tests -v
```

Render 部署入口：

- [一键导入 Render](https://render.com/deploy?repo=https://github.com/xiaoxinw50-debug/AIGC-)

## 发布方式

在 GitHub 仓库设置中开启：

- Branch: `main`
- Folder: `/ (root)`

然后直接访问仓库对应的 Pages 地址即可。
