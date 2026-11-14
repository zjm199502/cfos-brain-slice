; FosMap 脑片活动图谱（原名 cFos 脑片分析）Windows 安装程序（NSIS）。由 packaging/build_windows.sh 生成 stage 目录后编译。
Unicode true
!define APPNAME "cFos脑片分析"     ; 安装目录和注册表键沿用旧名：旧版升级、补丁手动安装脚本都靠它找到安装位置
!define DISPNAME "FosMap 脑片活动图谱"
!define OLDDISP "cFos 脑片分析"
!define VERSION "0.8.1"
!ifndef STAGE
  !define STAGE "stage"
!endif
!ifndef OUTFILE
  !define OUTFILE "FosMap脑片活动图谱_v0.8.1_Windows安装包.exe"
!endif

Name "${DISPNAME} ${VERSION}"
OutFile "${OUTFILE}"
; 装到当前用户的 AppData\Local 下：不需要管理员权限，软件也能写入自己的文件夹
InstallDir "$LOCALAPPDATA\${APPNAME}"
InstallDirRegKey HKCU "Software\${APPNAME}" "InstallDir"
RequestExecutionLevel user
SetCompressor lzma
SetCompressorDictSize 64
ManifestDPIAware true

!include "MUI2.nsh"
!define MUI_ICON "${STAGE}\app\cfos.ico"
!define MUI_UNICON "${STAGE}\app\cfos.ico"
!define MUI_WELCOMEPAGE_TITLE "欢迎安装 ${DISPNAME} ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "这个程序会把 ${DISPNAME} 安装到你的电脑上。$\r$\n$\r$\n安装不需要联网，也不需要管理员权限，约需 4 GB 磁盘空间、3–8 分钟。$\r$\n$\r$\n软件只使用 CPU，不需要独立显卡。"
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchApp
!define MUI_FINISHPAGE_RUN_TEXT "现在启动 ${DISPNAME}"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"

Function LaunchApp
  SetOutPath "$INSTDIR\app"
  Exec '"$INSTDIR\python\pythonw.exe" "$INSTDIR\app\main.py"'
FunctionEnd

Section "主程序" SecMain
  SectionIn RO
  ; 升级安装：先删掉旧的程序文件（项目和结果不在这里，不受影响）
  RMDir /r "$INSTDIR\app\cfos"
  SetOutPath "$INSTDIR\python"
  File /r "${STAGE}\python\*.*"
  SetOutPath "$INSTDIR\app"
  File /r "${STAGE}\app\*.*"
  File "${STAGE}\诊断启动（出问题时用）.bat"

  DetailPrint "正在预编译程序文件，让以后启动更快（约 1–3 分钟）…"
  nsExec::ExecToLog '"$INSTDIR\python\python.exe" -m compileall -q -j 0 "$INSTDIR\app\cfos" "$INSTDIR\python\Lib"'
  Pop $0

  ; 改名前（cFos 脑片分析）的快捷方式
  Delete "$DESKTOP\${OLDDISP}.lnk"
  RMDir /r "$SMPROGRAMS\${OLDDISP}"
  SetOutPath "$INSTDIR\app"
  CreateShortCut "$DESKTOP\${DISPNAME}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\main.py"' "$INSTDIR\app\cfos.ico"
  CreateDirectory "$SMPROGRAMS\${DISPNAME}"
  CreateShortCut "$SMPROGRAMS\${DISPNAME}\${DISPNAME}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\main.py"' "$INSTDIR\app\cfos.ico"
  CreateShortCut "$SMPROGRAMS\${DISPNAME}\诊断启动（出问题时用）.lnk" "$INSTDIR\app\诊断启动（出问题时用）.bat" "" "$INSTDIR\app\cfos.ico"
  CreateShortCut "$SMPROGRAMS\${DISPNAME}\卸载 ${DISPNAME}.lnk" "$INSTDIR\uninstall.exe"

  WriteRegStr HKCU "Software\${APPNAME}" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${DISPNAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayIcon" "$INSTDIR\app\cfos.ico"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "NoModify" 1
  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\${DISPNAME}.lnk"
  Delete "$DESKTOP\${OLDDISP}.lnk"
  RMDir /r "$SMPROGRAMS\${OLDDISP}"
  RMDir /r "$SMPROGRAMS\${DISPNAME}"
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\app"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
  DeleteRegKey HKCU "Software\${APPNAME}"
SectionEnd
