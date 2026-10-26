"""Regenerate all notebooks from the single library sources in this folder.
Usage: python tools/build_all.py   (does not execute the notebooks)"""
import os, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
for b in ['build_d4val.py', 'build_msdval.py', 'build_analysis.py']:
    runpy.run_path(os.path.join(HERE, b), run_name='__main__')
