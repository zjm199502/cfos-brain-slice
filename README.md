# cFos 脑片分析 安装包（v0.8）

## 已经装了 v0.8 或更新版本：不用下载这里的文件
软件启动时会自动检查更新，有新版本时显示更新公告，点“现在更新”即可（只下载约 0.2 MB 的补丁）。
也可以在 `updates/` 里手动下载补丁 zip，在软件里点“帮助 → 从文件安装补丁…”。所有版本的更新公告见 [updates/更新公告.md](updates/更新公告.md)。

## 还在用 v0.7：两种升级方式任选
- **只下载补丁（约 0.2 MB）**：下载 `updates/cfos-patch-0.8.zip`，关闭软件，解压后 Windows 双击“手动安装（Windows 旧版本用）.bat”，Mac 右键点“手动安装（Mac 旧版本用）.command”→“打开”。之后的更新就能在软件里自动完成。
- **重新安装完整安装包**（下面）。

## 完整安装包（第一次安装用）
GitHub 网页上单个文件不能超过 100 MB，所以安装包分成了若干个 90 MB 的分卷。把同一个文件夹里的全部分卷下载到同一处后合并：

- **Windows**（`windows/`，8 个分卷）：下载全部 `.partNN` 和 `合并.bat` 到同一个文件夹，双击 `合并.bat`，得到 `cFos-v0.8-Windows-setup.exe`（710 MB）。MD5 见 `windows/MD5.txt`。
- **Mac**（`mac/`，10 个分卷，Apple M 系列芯片，macOS 15 以上）：在终端中进入下载文件夹，运行
  `cat cFos-v0.8-Mac-AppleSilicon.zip.part* > cFos-v0.8-Mac-AppleSilicon.zip`
  得到 901 MB 的 zip，MD5 见 `mac/MD5.txt`。解压后把“cFos脑片分析”拖到“应用程序”；第一次打开若提示“无法验证开发者”，到“系统设置 → 隐私与安全性”点“仍要打开”。

安装后请先在“① 项目与图片”页面点“软件自检”，看到“全部正常”即可。出问题时点“导出诊断包”，把 zip 发给开发者。

源码在 `main` 分支；用到的第三方软件包、模型和数据见 main 分支的 THIRD_PARTY.md。

联系方式：张金铭，zjm199502@163.com，https://github.com/zjm199502/cfos-brain-slice
