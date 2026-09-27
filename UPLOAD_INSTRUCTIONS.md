# 上传说明

## 需要覆盖或新增的文件

| 仓库路径 | 操作 |
|---|---|
| `app.py` | 覆盖 |
| `report_verification.py` | 新增 |
| `render.yaml` | 覆盖 |
| `webapp/templates/index.html` | 覆盖 |
| `webapp/templates/verify.html` | 新增 |
| `webapp/static/app.js` | 覆盖 |
| `tests/test_app.py` | 覆盖 |
| `tests/test_frontend_state.cjs` | 覆盖 |
| `tests/test_report_verification.py` | 新增 |

请保留上表的目录层级，不要只把所有文件放到仓库根目录。

## Render 必做配置

`render.yaml` 已经声明以下环境变量：

- `AIGC_REPORT_SIGNING_KEY`：自动生成的报告签名密钥。
- `AIGC_REPORT_ISSUER`：北京大学文化数据治理实验室 AIGC 标识治理项目。
- `AIGC_PUBLIC_BASE_URL`：`https://aigc-image-inference-demo.onrender.com`。

如果当前 Render 服务不是通过 Blueprint 管理，单独修改 `render.yaml` 可能不会自动写入已存在服务的环境变量。此时需在 Render 的 Environment 中手动新增这三项。`AIGC_REPORT_SIGNING_KEY` 请使用至少 32 字节的随机值，不要写入 GitHub 文件。

## 部署后检查

1. 打开 `/api/health`，确认 `report_verification.signing_ready` 为 `true`。
2. 上传一张图片，导出单图报告，确认显示“签名有效，可在线验证”。
3. 点击“在线验证报告”，确认验证页显示相同的报告编号。
4. 如果页面仍为旧版，确认 `index.html` 中的静态资源版本为 `laboratory-20260927-verified-report-v1`。

## 本地测试结果

- Python：7 项通过。
- Node：16 项通过。
- 实际签发和验签：通过。
- 篡改凭证拒绝：通过。
