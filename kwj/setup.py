from http.server import executable
from cx_Freeze import setup, Executable

setup(name = "Woojin_Object_Detection_Software",
    version = "0.1",
    description = "This software detects objects in realtime",
    executable = [Executable("detect.py")]
)