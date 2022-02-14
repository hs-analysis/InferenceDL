from setuptools import setup, find_packages
import pkg_resources
import pathlib


setup(
  name = 'comdl',
  packages = find_packages(),
  version = '0.0.1',
  install_requires=[],
  license='MIT',
  description = 'ComDL - pytorch',
  author = 'Philipp Marquardt',
  author_email = 'p.marquardt15@googlemail.com',
  url = 'https://github.com/PhilippMarquardt/InferenceDL',
  package_data={'': ['*.yaml']}
)