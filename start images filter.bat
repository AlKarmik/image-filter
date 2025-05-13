@echo off
rem ----------------- run_filter.bat -----------------
rem Фильтрация 200k аниме-кадров на GPU

:: 1) переходим на диск и в корень, где лежат venv и скрипт
F:
cd \

:: 2) активируем виртуальное окружение
call F:\wd14_env\Scripts\activate

:: 3) запускаем фильтр
python fast_filter_minors.py --src "F:\photos" --dst "F:\adult_only" --batch 128 --thr 0.4

:: 4) пауза, чтобы увидеть итог
echo.
echo -------- Done --------
pause
