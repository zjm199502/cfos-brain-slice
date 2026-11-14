# FosMap 脑片活动图谱（原名 cFos 脑片分析）安装包（完整安装包 v0.8，可自动更新到 v0.8.2）

**使用说明书**（研一新生版，含安装步骤和每一步的截图）：`cFos脑片分析_使用说明书_v0.7.pdf`（直接在网页上看）或 `cFos脑片分析_使用说明书_v0.7.docx`（Word）。v0.8 的操作和 v0.7 相同，新增功能见更新公告。

## 已经装了 v0.8 或更新版本：不用下载这里的文件
软件启动时会自动检查更新，有新版本时显示更新公告，点“现在更新”即可（只下载约 0.3 MB 的补丁）。
也可以在 `updates/` 里手动下载补丁 zip，在软件里点“帮助 → 从文件安装补丁…”。所有版本的更新公告见 [updates/更新公告.md](updates/更新公告.md)。

## 还在用 v0.7：两种升级方式任选
- **只下载补丁（约 0.3 MB）**：下载 `updates/cfos-patch-0.8.2.zip`，关闭软件，解压后 Windows 双击“手动安装（Windows 旧版本用）.bat”，Mac 右键点“手动安装（Mac 旧版本用）.command”→“打开”。之后的更新就能在软件里自动完成。
- **重新安装完整安装包**（下面）。

## 完整安装包（第一次安装用）
**推荐：到 [Release v0.8](https://github.com/zjm199502/cfos-brain-slice/releases/tag/v0.8) 直接下载单个文件**（不用合并）：
- Mac：[cFos-v0.8-Mac-AppleSilicon.zip](https://github.com/zjm199502/cfos-brain-slice/releases/download/v0.8/cFos-v0.8-Mac-AppleSilicon.zip)（901 MB），下载后从下面 Mac 步骤的第 3 步开始。
- Windows：上传后会出现在同一个 Release 页面（文件名 `cFos-v0.8-Windows-setup.exe`）。

下载 Release 不方便时，也可以用下面的分卷：
GitHub 仓库里单个文件不能超过 100 MB，所以安装包分成了若干个 90 MB 的分卷。**不用下载整个分支**：点下面的链接逐个下载（每个链接直接下载一个分卷），放到同一个文件夹里再合并。

### Windows（8 个分卷 + 合并.bat，合并后 710 MB）
  - [MD5.txt](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/MD5.txt)
  - [cFos-v0.8-Windows-setup.exe.part01](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/cFos-v0.8-Windows-setup.exe.part01)
  - [cFos-v0.8-Windows-setup.exe.part02](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/cFos-v0.8-Windows-setup.exe.part02)
  - [cFos-v0.8-Windows-setup.exe.part03](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/cFos-v0.8-Windows-setup.exe.part03)
  - [cFos-v0.8-Windows-setup.exe.part04](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/cFos-v0.8-Windows-setup.exe.part04)
  - [cFos-v0.8-Windows-setup.exe.part05](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/cFos-v0.8-Windows-setup.exe.part05)
  - [cFos-v0.8-Windows-setup.exe.part06](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/cFos-v0.8-Windows-setup.exe.part06)
  - [cFos-v0.8-Windows-setup.exe.part07](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/cFos-v0.8-Windows-setup.exe.part07)
  - [cFos-v0.8-Windows-setup.exe.part08](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/cFos-v0.8-Windows-setup.exe.part08)
  - [合并.bat](https://github.com/zjm199502/cfos-brain-slice/raw/installers/windows/%E5%90%88%E5%B9%B6.bat)

下载后双击 `合并.bat`，得到 `cFos-v0.8-Windows-setup.exe`，双击安装（不需要管理员权限）。

### Mac（Apple M 系列芯片，macOS 15 以上；10 个分卷，合并后 901 MB）
  - [MD5.txt](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/MD5.txt)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part01](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part01)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part02](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part02)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part03](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part03)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part04](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part04)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part05](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part05)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part06](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part06)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part07](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part07)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part08](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part08)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part09](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part09)
  - [cFos-v0.8-Mac-AppleSilicon.zip.part10](https://github.com/zjm199502/cfos-brain-slice/raw/installers/mac/cFos-v0.8-Mac-AppleSilicon.zip.part10)

1. 把 10 个分卷都下载到“下载”文件夹。
2. 打开“终端”（启动台 → 其他 → 终端），粘贴下面这行并回车：
   `cd ~/Downloads && cat cFos-v0.8-Mac-AppleSilicon.zip.part* > cFos-v0.8-Mac-AppleSilicon.zip`
3. 双击得到的 `cFos-v0.8-Mac-AppleSilicon.zip` 解压，把“cFos脑片分析”拖到“应用程序”文件夹。
4. 第一次打开：在“应用程序”里双击“cFos脑片分析”。如果提示“无法验证开发者”或“已损坏”，点“完成/取消”，打开“系统设置 → 隐私与安全性”，在页面下方点“仍要打开”，再输入开机密码确认。以后就能正常双击打开。
5. 如果第 4 步仍提示“已损坏”，在终端运行：`xattr -dr com.apple.quarantine /Applications/cFos脑片分析.app`，再打开。

安装后请先在“① 项目与图片”页面点“软件自检”，看到“全部正常”即可。出问题时点“导出诊断包”，把 zip 发给开发者。

源码在 `main` 分支；用到的第三方软件包、模型和数据见 main 分支的 THIRD_PARTY.md。

联系方式：zjm199502@163.com，https://github.com/zjm199502/cfos-brain-slice
