#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from setuptools import setup, find_packages

try:
    README = open('README.md').read()
except Exception:
    README = ''
VERSION = "1.0.0"
requirments = ["pytest==4.4.1",
               "pysenal==0.0.2",
               "sklearn_crfsuite==0.3.6",
               "modelhub==3.0.3",
               "pandas==0.24.2",
               "intervaltree==3.0.2",
               "setuptools==41.0.0",
               "joblib==0.13.2",
               "PyYAML==5.1.1",
               "spacy==2.1.4, en_core_web_sm==2.1.0"]

setup(
    name='datagrand_ie_2019',
    version=VERSION,
    description='datagrand_ie_2019',
    url="https://github.com/supercoderhawk/datagrand-ie-2019",
    long_description=README,
    author='yubin.xia',
    author_email='xiayubin@patsnap.com',
    packages=find_packages(exclude=('notebooks', 'tests', 'scripts')),
    install_requires=requirments,
    extras_require={
    },
)
