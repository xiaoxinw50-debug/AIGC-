# 文化数据治理实验室页面部署说明

## 部署目标

- AI 图片识别：`https://aigc-image-inference-demo.onrender.com/`
- 实验室页面：`https://aigc-image-inference-demo.onrender.com/lab/`
- 健康检查：`https://aigc-image-inference-demo.onrender.com/api/health`

本上传包基于 GitHub `main` 分支提交 `324b3b44c699562ec978876fc96ae4a168b6378a` 合并生成，保留现有签名报告核验等线上功能。

## 上传方式

将上传包内文件按原目录覆盖或新增到 GitHub 仓库根目录：

```text
app.py
README.md
DEPLOY_LAB.md
tests/test_app.py
lab_site/
```

提交到 `main` 后，Render Auto-Deploy 会继续使用现有 `render.yaml` 构建，无需新建服务或新增环境变量。

## 发布后检查

1. 打开 `/api/health`，确认 `lab_site.available` 为 `true`。
2. 打开 `/lab/`，确认首屏、导航和移动端菜单正常。
3. 打开 `/`，确认图片识别和签名报告核验仍可使用。
