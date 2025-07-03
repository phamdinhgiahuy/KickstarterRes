@echo off
echo Kickstarter Scraper Utilities
echo ==============================
echo.
echo 1. Check cache status
echo 2. Rebuild cache from existing projects
echo 3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Checking cache status...
    python check_cache_status.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo Rebuilding cache...
    python rebuild_cache.py
    pause
) else if "%choice%"=="3" (
    exit
) else (
    echo Invalid choice. Please try again.
    pause
    goto :eof
)

echo.
echo Operation completed.
pause
