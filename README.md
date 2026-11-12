# cFos 脑片分析 安装包（v0.5.1）

GitHub 网页上单个文件不能超过 100 MB，所以安装包分成了若干个 90 MB 的分卷。把同一个文件夹里的全部分卷下载到同一处后合并：

- **Windows**（`windows/`，9 个分卷）：下载全部 `.partNN` 和 `合并.bat` 到同一个文件夹，双击 `合并.bat`，得到 `cFos-v0.5.1-Windows-setup.exe`。MD5 见 `windows/MD5.txt`。
- **Mac**（`mac/`，12 个分卷，Apple M 系列芯片，macOS 15 以上）：在终端中进入下载文件夹，运行
  `cat cFos-v0.5.1-Mac-AppleSilicon.zip.part* > cFos-v0.5.1-Mac-AppleSilicon.zip`
  MD5 48da90d396f8ef49b834b5570e8129c1。

v0.5.1 修复了多核 Windows 电脑上细胞识别极慢的问题（关闭 TensorFlow 的 oneDNN），并且最多只用一半的 CPU 核。源码在 `main` 分支。
