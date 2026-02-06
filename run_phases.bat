@echo off
set BASE_NAME=AsesAgent
set EPISODES=5000

rem echo ========================================================
rem echo STARTED: Phase 1 (Aiming Mastery)
rem echo Model: %BASE_NAME%_Phase1
rem echo ========================================================
rem python src/controller/trainer.py --only-phase 1 --episodes %EPISODES% --model %BASE_NAME%_Phase1 --fast
rem if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ========================================================
echo STARTED: Phase 2 (One Shot Mode - ELITE)
echo Model: %BASE_NAME%_Phase2 (Loading from Phase1)
echo ========================================================
python src/controller/trainer.py --only-phase 2 --episodes 10000 --model %BASE_NAME%_Phase2 --load-model %BASE_NAME%_Phase1 --min-success 0.9 --fast
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ========================================================
echo STARTED: Phase 3 (IFF & Modern Warfare)
echo Model: %BASE_NAME%_Phase3 (Loading from Phase2)
echo ========================================================
python src/controller/trainer.py --only-phase 3 --episodes 10000 --model %BASE_NAME%_Phase3 --load-model %BASE_NAME%_Phase2 --min-success 0.8 --fast
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ========================================================
echo STARTED: Phase 4 (Full War Mode)
echo Model: %BASE_NAME%_Phase4 (Loading from Phase3)
echo ========================================================
python src/controller/trainer.py --only-phase 4 --episodes 5000 --model %BASE_NAME%_Phase4 --load-model %BASE_NAME%_Phase3 --min-success 0.8 --fast
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ========================================================
echo TRAINING COMPLETE!
echo Final Model: models/%BASE_NAME%_Phase4
echo ========================================================
pause
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo ERROR: Training failed or was interrupted.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pause
exit /b 1
