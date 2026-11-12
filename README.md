# cFos 脑片分析 安装包（v0.5）

GitHub 网页上单个文件不能超过 100 MB，所以安装包分成了若干个 90 MB 的分卷。下载同一个文件夹里的全部分卷后合并：

- **Windows**（`windows/`，9 个分卷）：下载全部 `.partNN` 和 `合并.bat` 到同一个文件夹，双击 `合并.bat`，得到 `cFos-v0.5-Windows-setup.exe`（843,416,564 字节，MD5 c8a32dcd706b6f20b1b81c74008c325d）。
- **Mac**（`mac/`）：在终端中进入下载文件夹，运行 `cat cFos-v0.5-Mac-AppleSilicon.zip.part* > cFos-v0.5-Mac-AppleSilicon.zip`（MD5 58a1c7de9cdc1afaa73504881f49fbcd）。

源码在 `main` 分支。
