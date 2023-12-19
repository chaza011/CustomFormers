from setuptools import setup, find_packages

setup(
    name='CustomFormers',
    version='0.1',
    url = 'https://github.com/chaza011/CustomFormers',
    packages='CustomFormers.Modules',
    install_requires=[
        'torch>=1.13',
    ]
)


