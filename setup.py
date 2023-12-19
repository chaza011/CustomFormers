from setuptools import setup, find_packages

setup(
    name='CustomFormers',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.13',
    ],
)


