@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在以诊断模式启动 FosMap 脑片活动图谱，出错信息会显示在这个窗口里。
echo 如果软件打不开，请把这个窗口截图发给开发者。
"%~dp0..\python\python.exe" "%~dp0main.py" %*
echo.
echo 软件已退出（代码 %errorlevel%）。
pause
