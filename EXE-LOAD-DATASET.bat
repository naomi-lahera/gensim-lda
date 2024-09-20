@echo off
REM Obtener la ruta absoluta del directorio donde se ejecuta el archivo .bat
set "abs_path=%~dp0"

REM Definir la ruta del archivo JSON y el script Python usando la ruta absoluta
set "config_file=%abs_path%config.json"
set "python_file=%abs_path%src\dataset.py"

REM Mostrar las rutas obtenidas (opcional para depuración)
echo config path: %config_file%
echo file: %python_file%
echo wait...

REM Ejecutar el script de Python con el archivo JSON como argumento
python "%python_file%" "%config_file%"

REM Mantener la consola abierta hasta que el usuario la cierre manualmente
echo.
echo Presiona cualquier tecla para cerrar la ventana...
pause >nul
