@echo off
REM Run this ON THE WINDOWS MACHINE, inside the depth_calib_tool folder.
REM Produces a folder dist\DepthCalibTool\ containing DepthCalibTool.exe
REM plus its dependencies. Copy that whole folder to share with a partner
REM (not just the .exe -- --onedir mode needs the folder alongside it).

python -m venv build_env
call build_env\Scripts\activate.bat

pip install -r requirements.txt
pip install pyinstaller

pyinstaller --onedir --noconsole --name DepthCalibTool ^
  --collect-all torch ^
  --collect-all transformers ^
  --collect-all timm ^
  --collect-all cv2 ^
  --collect-all gradio ^
  --collect-all gradio_client ^
  --collect-all matplotlib ^
  --collect-all segment_anything ^
  app.py

echo.
echo Build complete. See dist\DepthCalibTool\
echo Zip that whole folder to send to your partner.
pause
