"""
Minimal setup.py so `pip install -e .` makes the project root
importable as a package from anywhere inside the venv.

After running `pip install -e .` once, you no longer need PYTHONPATH=.
and can simply do: python ingestion/video_processor.py
"""

from setuptools import setup, find_packages

setup(
    name="eduvision-rag",
    version="0.1.0",
    packages=find_packages(exclude=["venv", "venv.*", "data", "videos"]),
)
