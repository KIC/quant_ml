from setuptools import setup

setup(
   name='talib-ml',
   version='0.0.0',
   author='KIC',
   author_email='',
   packages=['talib_ml'],
   scripts=[],
   url='',
   license='',
   description='An awesome package that does something',
   long_description='', # open('README.txt').read(),
   install_requires=open("requirements.txt").read().splitlines(),
)