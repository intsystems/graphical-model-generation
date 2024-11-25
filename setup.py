import io
import re
from setuptools import setup, find_packages

from code import __version__


def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('docs/index.md')
# clearing local versions
# https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers

requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$', read('utils/requirements.txt'),
               flags=re.MULTILINE))

setup(
    # metadata
    name='GMG_auto',
    version=__version__,
    license='MIT',
    author='Ernest Nasyrov',
    author_email="nasyrov.rr@phystech.edu",
    description='GMG_auto, python package',
    long_description=readme,
    url='https://github.com/intsystems/graphical-model-generation',
    # options
    packages=find_packages(),
    install_requires=requirements,
)