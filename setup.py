from setuptools import setup, find_namespace_packages

setup(
   name='quant-ml',
   version='0.0.0',
   author='KIC',
   author_email='',
   packages=find_namespace_packages(include=["quant_ml.*"]),
   scripts=[],
   url='https://github.com/KIC/quant_ml',
   license='MIT',
   description='re-implementation of TA-Lib suitable to pandas-ml-utils',
   long_description=open('Readme.md').read(),
   install_requires=open("requirements.txt").read().splitlines(),
   extras_require={
      "dev": open("dev-requirements.txt").read().splitlines(),
   },
)