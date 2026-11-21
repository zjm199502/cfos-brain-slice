# FosFolio 脑片图册安装与更新

## 安装 v0.9.0

升级前请退出正在运行的软件。v0.8.x 用户可以从「帮助 → 检查更新 → 打开下载页面」进入同一发布页。[v0.9.0 正式发布页](https://github.com/zjm199502/cfos-brain-slice/releases/tag/v0.9.0)

| 平台 | 下载及安装 |
| --- | --- |
| Windows x64 | 双击运行 [FosFolio-0.9.0-Windows-x64-Setup.exe](https://github.com/zjm199502/cfos-brain-slice/releases/download/v0.9.0/FosFolio-0.9.0-Windows-x64-Setup.exe)，按安装向导操作。 |
| Apple 芯片 Mac（macOS 15 及以上） | 解压 [FosFolio-0.9.0-macOS-arm64.zip](https://github.com/zjm199502/cfos-brain-slice/releases/download/v0.9.0/FosFolio-0.9.0-macOS-arm64.zip)，将应用拖入“应用程序”文件夹。 |

首次识别前，进入「④ 细胞识别」，点击「安装 / 修复 Cellpose 4 运行环境」。请保持联网并预留数 GB 磁盘空间，模型下载约 1.23 GB。完成后点「测试 Cellpose 设备」，查看实际使用的 CUDA、MPS 或 CPU。以后的识别可离线进行。

旧项目可以继续打开并查看已有计数。使用 v0.9.0 重新识别时，软件会对选中的脑片生成 Cellpose 4 结果。[历次更新公告](updates/更新公告.md)

## 以往版本

本分支的 `windows/`、`mac/` 目录保存 v0.8 分卷，`updates/` 保存 v0.8.x 历史补丁。如需恢复旧环境，可查看 [v0.8 安装说明](https://github.com/zjm199502/cfos-brain-slice/blob/5604f08b003489e9435afa5554070770fb044f3c/README.md)。
