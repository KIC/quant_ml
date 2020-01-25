from setuptools import setup

setup(
   name='quant-ml',
   version='0.0.0',
   author='KIC',
   author_email='',
   packages=['quant_ml'],
   scripts=[],
   url='https://github.com/KIC/quant_ml',
   license='MIT',
   description='re-implementation of TA-Lib suitable to pandas-ml-utils',
   long_description=open('README.txt').read(),
   install_requires=open("requirements.txt").read().splitlines(),
   extras_require={
      "dev": open("dev-requirements.txt").read().splitlines(),
   },
)