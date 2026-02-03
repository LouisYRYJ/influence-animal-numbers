# pyright: standard
import sys
import os
from os import environ

home = environ.get("HOME")
run_folder = os.path.dirname(os.path.abspath(__file__))
runs_folder = os.path.abspath(os.path.join(run_folder, ".."))
project_folder = os.path.abspath(os.path.join(runs_folder, ".."))

sys.path.insert(0, os.path.join(
    runs_folder, "common"
))

sys.path.insert(0, os.path.join(
    project_folder, "finetuning"
))