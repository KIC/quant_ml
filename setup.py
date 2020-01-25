from setuptools import setup

setup(
   name='quant-ml',
   version='0.0.0',
   author='KIC',
   author_email='',
   packages=['quant_ml'],
   scripts=[],
   url='',
   license='',
   description='re-implementation of TA-Lib suitable to pandas-ml-utils',
   long_description='', # open('README.txt').read(),
   install_requires=open("requirements.txt").read().splitlines(),
)