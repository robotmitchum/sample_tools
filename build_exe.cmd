REM Build executable on Windows
SET DIR=%~dp0
cd %DIR%
call %DIR%\.venv\Scripts\activate.bat
python -m build_exe ir_tool_UI.py build_config.json bundle_config.json
pause