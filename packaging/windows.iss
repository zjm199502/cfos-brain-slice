; Native Windows build. Run packaging/build_windows.ps1 with a vetted Python x64 runtime and the release resources.
#define ProductName "FosFolio 脑片图册"
#define LegacyId "cFos脑片分析"

[Setup]
AppId={code:ApplicationId}
AppName={#ProductName}
AppVersion={#AppVersion}
VersionInfoVersion={#AppVersion}.0
AppPublisher=FosFolio
AppPublisherURL=https://github.com/zjm199502/cfos-brain-slice
DefaultDirName={code:InstallDirectory}
DefaultGroupName={#ProductName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
DisableProgramGroupPage=yes
DisableDirPage=no
UsePreviousLanguage=no
WizardStyle=modern
#ifdef CellposeBundle
OutputBaseFilename=FosFolio-{#AppVersion}-Windows-x64-Offline-Setup
DiskSpanning=yes
DiskSliceSize=2000000000
#else
OutputBaseFilename=FosFolio-{#AppVersion}-Windows-x64-Setup
#endif
SetupIconFile={#Stage}\app\cfos.ico
UninstallDisplayIcon={app}\app\cfos.ico
Compression=lzma2/fast
SolidCompression=yes
LZMANumBlockThreads=4
CloseApplications=yes
RestartApplications=no
ChangesAssociations=no

[Languages]
Name: "zh"; MessagesFile: "ChineseSimplified.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
zh.Launch=启动 FosFolio 脑片图册
en.Launch=Open FosFolio
zh.Diagnostic=诊断启动（出问题时用）
en.Diagnostic=Diagnostic startup
zh.DesktopShortcut=创建桌面快捷方式
en.DesktopShortcut=Create a desktop shortcut
#ifdef CellposeBundle
zh.CellposeInstalling=正在从安装包配置 Cellpose 4 与模型并进行推理测试，首次安装可能需要数分钟…
en.CellposeInstalling=Installing Cellpose 4 and its model from local media and running an inference test. This can take several minutes.
zh.CellposeFailure=FosFolio 已安装。Cellpose 4 离线配置失败。请在软件的“④ 细胞识别”中重新安装或修复 Cellpose 4 运行环境。
en.CellposeFailure=FosFolio is installed. Cellpose 4 offline setup failed. Open Cell Detection in FosFolio and use Install / Repair Cellpose 4 to retry from the installed files.
#endif

[Tasks]
Name: desktopicon; Description: "{cm:DesktopShortcut}"

[Files]
Source: "{#Stage}\python\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#Stage}\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs createallsubdirs ignoreversion
#ifdef CellposeBundle
Source: "{#CellposeBundle}\*"; DestDir: "{app}\app\resources\cellpose4"; Flags: recursesubdirs ignoreversion nocompression
#endif

[Icons]
Name: "{code:StartMenuFolder}\{#ProductName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\cfos.ico"
Name: "{code:StartMenuFolder}\{cm:Diagnostic}"; Filename: "{app}\app\诊断启动（出问题时用）.bat"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\cfos.ico"
Name: "{code:StartMenuFolder}\卸载 {#ProductName}"; Filename: "{uninstallexe}"
Name: "{code:DesktopFolder}\{#ProductName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\cfos.ico"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\{code:ApplicationId}"; ValueType: string; ValueName: "InstallDir"; ValueData: "{app}"; Flags: uninsdeletevalue uninsdeletekeyifempty

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; Description: "{cm:Launch}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Python creates caches at runtime. Application projects, components and update history live outside these directories.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app\cfos"
Type: dirifempty; Name: "{app}\app"

[Code]
#ifdef CellposeBundle
var
  CellposeFailed: Boolean;

procedure SetupCellpose;
var
  ResultCode: Integer;
  AppDir, PythonPath: String;
begin
  AppDir := ExpandConstant('{app}\app');
  PythonPath := ExpandConstant('{app}\python\python.exe');
  WizardForm.StatusLabel.Caption := CustomMessage('CellposeInstalling');
  CellposeFailed := True;
  try
    if ExecAndLogOutput(PythonPath, '-u "' + AppDir + '\main.py" --install-cellpose-offline',
        AppDir, SW_HIDE, ewWaitUntilTerminated, ResultCode, nil) then
      CellposeFailed := ResultCode <> 0;
    Log(Format('Cellpose offline setup result: %d', [ResultCode]));
  except
    Log(GetExceptionMessage);
  end;
  if CellposeFailed then
    SuppressibleMsgBox(CustomMessage('CellposeFailure'), mbError, MB_OK, IDOK);
end;

function GetCustomSetupExitCode: Integer;
begin
  if CellposeFailed then Result := 10 else Result := 0;
end;
#endif

function AcceptanceMode: Boolean;
begin
  Result := ExpandConstant('{param:FOSFOLIOTEST|}') = '1';
end;

function ApplicationId(Param: String): String;
begin
  Result := '{#LegacyId}';
  if AcceptanceMode then Result := Result + '-InstallerQA';
end;

function InstallDirectory(Param: String): String;
begin
  if not RegQueryStringValue(HKCU, 'Software\' + ApplicationId(''), 'InstallDir', Result) then
    Result := ExpandConstant('{localappdata}\') + ApplicationId('');
end;

function StartMenuFolder(Param: String): String;
begin
  if AcceptanceMode then Result := ExpandConstant('{app}\test-shell\StartMenu\{#ProductName}')
  else Result := ExpandConstant('{userprograms}\{#ProductName}');
end;

function DesktopFolder(Param: String): String;
begin
  if AcceptanceMode then Result := ExpandConstant('{app}\test-shell\Desktop')
  else Result := ExpandConstant('{userdesktop}');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var Key, LegacyCommand, LegacyUninstaller: String;
begin
  if CurStep <> ssPostInstall then exit;
  { Migrate the previous NSIS uninstall registration only when it owns this exact folder.
    The new Inno registration is already present and uses the same ID with its standard _is1 suffix. }
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\' + ApplicationId('');
  LegacyUninstaller := ExpandConstant('{app}\uninstall.exe');
  if RegQueryStringValue(HKCU, Key, 'UninstallString', LegacyCommand) and
     (CompareText(RemoveQuotes(LegacyCommand), LegacyUninstaller) = 0) then begin
    RegDeleteKeyIncludingSubkeys(HKCU, Key);
    DeleteFile(LegacyUninstaller);
  end;
#ifdef CellposeBundle
  SetupCellpose;
#endif
end;
