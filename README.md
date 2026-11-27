# FosFolio 脑片图册安装与更新

## 当前版本 0.9.22

已安装 0.9.x 的用户可打开「帮助 → 检查更新 → 现在更新」，安装累计补丁后重启。

使用 0.7 或 0.8 运行环境的用户，请先安装 [Windows x64 基础版](https://github.com/zjm199502/cfos-brain-slice/releases/download/v0.9.0/FosFolio-0.9.0-Windows-x64-Setup.exe) 或 [Apple 芯片 Mac 基础版](https://github.com/zjm199502/cfos-brain-slice/releases/download/v0.9.0/FosFolio-0.9.0-macOS-arm64.zip)，再在软件内检查更新至 0.9.22。[0.9.22 累计补丁](https://raw.githubusercontent.com/zjm199502/cfos-brain-slice/installers/updates/v0.9/cfos-patch-0.9.22.zip)

继续使用 StarDist 0.8 系列的用户可安装 [0.8.12 兼容补丁](https://raw.githubusercontent.com/zjm199502/cfos-brain-slice/installers/updates/v0.8/cfos-patch-0.8.12.zip)。此补丁提供「帮助 → 升级至 FosFolio 0.9.19…」入口。

## 新安装 v0.9.0（Cellpose 4）

国内用户推荐优先从腾讯云下载离线包，内含 Cellpose 4 运行环境和模型。此完整安装包会切换识别引擎。[腾讯云 CNB 下载](https://cnb.cool/fosfolio/cfos-brain-slice/-/releases/tag/v0.9.0) · [GitHub 发布页](https://github.com/zjm199502/cfos-brain-slice/releases/tag/v0.9.0)

需要首次联网配置识别环境时，也可选择下列常规安装包。

| 平台 | 下载及安装 |
| --- | --- |
| Windows x64 | 双击运行 [FosFolio-0.9.0-Windows-x64-Setup.exe](https://github.com/zjm199502/cfos-brain-slice/releases/download/v0.9.0/FosFolio-0.9.0-Windows-x64-Setup.exe)，按安装向导操作。 |
| Apple 芯片 Mac（macOS 15 及以上） | 解压 [FosFolio-0.9.0-macOS-arm64.zip](https://github.com/zjm199502/cfos-brain-slice/releases/download/v0.9.0/FosFolio-0.9.0-macOS-arm64.zip)，将应用拖入“应用程序”文件夹。 |

首次识别前，进入「④ 细胞识别」，点击「安装 / 修复 Cellpose 4 运行环境」。请保持联网并预留数 GB 磁盘空间，模型下载约 1.23 GB。完成后点「测试 Cellpose 设备」，查看实际使用的 CUDA、MPS 或 CPU。以后的识别可离线进行。

旧项目可以继续打开并查看已有计数。使用 v0.9.0 重新识别时，软件会对选中的脑片生成 Cellpose 4 结果。[历次更新公告](updates/更新公告.md)

## 以往版本

本分支的 `windows/`、`mac/` 目录保存 v0.8 分卷，`updates/` 保存 v0.8.x 历史补丁。如需恢复旧环境，可查看 [v0.8 安装说明](https://github.com/zjm199502/cfos-brain-slice/blob/5604f08b003489e9435afa5554070770fb044f3c/README.md)。
