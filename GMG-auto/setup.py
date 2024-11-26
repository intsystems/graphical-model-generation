# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['gmg_auto']

package_data = \
{'': ['*']}

install_requires = \
['ipython>=7.34.0,<8.0.0',
 'openai>=1.55.1,<2.0.0',
 'pgmpy>=0.1.26,<0.2.0',
 'pydantic>=2.10.1,<3.0.0']

setup_kwargs = {
    'name': 'gmg-auto',
    'version': '0.1.9',
    'description': '',
    'long_description': '',
    'author': 'Ernest Nasyrov',
    'author_email': '2001092236@mail.ru',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}


setup(**setup_kwargs)

