# AI 图片识别与平台归因展示版

这是项目的展示与部署仓库，包含两部分：

1. `GitHub Pages` 静态展示页
2. `Render` 可推理 Flask 服务

## 说明

这个版本用于答辩和在线展示，保留了：

1. 网页原型的视觉结构
2. 典型样本切换
3. 平台归因结果展示
4. 文件层信号解释

## 静态展示版限制

GitHub Pages 不能运行 Flask / Python 后端，因此这个仓库是 **静态展示版**，不是在线实时推理系统。

## 可推理版本

仓库内已经提供 Render 所需文件：

- `render_app.py`
- `render.yaml`
- `requirements.txt`
- 模型自动还原模块 `model_payload.py`

可推理版本会提供：

1. 图片上传
2. `AI / real` 判断
3. 已采样平台归因
4. 文件层信号解释

Render 部署入口：

- [一键导入 Render](https://render.com/deploy?repo=https://github.com/xiaoxinw50-debug/AIGC-)

## 发布方式

在 GitHub 仓库设置中开启：

- Branch: `main`
- Folder: `/ (root)`

然后直接访问仓库对应的 Pages 地址即可。
