from pathlib import Path
from setuptools import setup, find_packages

project_root = Path(__file__).parent

install_requires = (project_root / 'requirements.txt').read_text().splitlines()

print(find_packages())

setup(
    name="mwzeval",
    author="Tomas Nekvinda",
    version="0.0.3",
    packages=find_packages(),
    package_data={'mwzeval': ['data/**/*.json', 'data/*.json']},
    python_requires=">=3.6",
    install_requires=install_requires,
) 
