# cFos 脑片分析 安装包（v0.6）

GitHub 网页上单个文件不能超过 100 MB，所以安装包分成了若干个 90 MB 的分卷。把同一个文件夹里的全部分卷下载到同一处后合并：

- **Windows**（`windows/`，9 个分卷）：下载全部 `.partNN` 和 `合并.bat` 到同一个文件夹，双击 `合并.bat`，得到 `cFos-v0.6-Windows-setup.exe`（843 MB）。MD5 见 `windows/MD5.txt`。
- **Mac**（`mac/`，12 个分卷，Apple M 系列芯片，macOS 15 以上）：在终端中进入下载文件夹，运行
  `cat cFos-v0.6-Mac-AppleSilicon.zip.part* > cFos-v0.6-Mac-AppleSilicon.zip`
  得到 1.06 GB 的 zip，MD5 见 `mac/MD5.txt`。解压后把“cFos脑片分析”拖到“应用程序”；第一次打开若提示“无法验证开发者”，到“系统设置 → 隐私与安全性”点“仍要打开”。

v0.6 新增：⑨ 截图制图（ROI 放大截图 + 可编辑比例尺，PNG/TIFF/SVG）；⑥ 配准中“拖动边界修正”（按住脑区边界线拖到正确位置，影响范围可调）。包含 v0.5.1 的多核 Windows 修复。源码在 `main` 分支。
