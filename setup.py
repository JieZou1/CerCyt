"""
The build/compilations setup

>> python setup.py install

"""
import pip
import logging
import pkg_resources
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path, session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
# try:
#     install_reqs = _parse_requirements("requirements.txt")
# except Exception:
#     logging.warning('Fail load requirements file, so using default ones.')
#     install_reqs = []

# For now, we just install CerCyt only. For missing packages, we will do it manually
install_reqs = []

setup(
    name='CerCyt',
    version='0.1',
    url='https://github.com/JieZou1/CerCyt',
    author='Jie Zou',
    author_email='jzou@mail.nlm.nih.gov',
    description='National Library of Medicine',
    packages=setuptools.find_packages(),
    install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.6',
    long_description=long_description,
    classifiers=[
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.6',
    ],
)
