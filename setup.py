from typing import List

import setuptools
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_install_requires() -> List[str]:
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets


setuptools.setup(
    name='fairclustering',
    version='0.0.8',
    author='Pascal Sigel',
    author_email='pascal.sigel@gmail.com',
    description='Fair clustering toolbox for python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/psigelo/fairclustering',
    license='MIT',
    packages=['fairclustering', "utils"],
    install_requires=get_install_requires(),
)