import os
import io
from setuptools import Command, find_packages, setup

DESCRIPTION="Interactive viewing of NetCDF data"

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/pmlmodelling/ncplot/issues",
    "Documentation": "https://ncplot.readthedocs.io/en/stable",
    "Source Code": "https://github.com/pmlmodelling/ncplot",
}

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

here = os.path.abspath(os.path.dirname(__file__))

# Use the README file as the description
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except IOError:
    long_description = DESCRIPTION

#extras_require: dict[str, list[str]] = {
#        ':python_version <= "3.10"': [
#            'datashader',
#        ],
#}

extras_require: dict() = {
        ':python_version <= "3.10"': [
            'datashader',
        ],
}


extras_require["complete"] = ["geoviews"]

setup(name='ncplot',
      version='0.4.2',
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',

      python_requires='>=3.6.1',

      entry_points={
        'console_scripts': [
            'ncplot =ncplot.command_line:main',
        ] },

      classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

      project_urls=PROJECT_URLS,
      url = "https://github.com/pmlmodelling/ncplot",
      author='Robert Wilson',
      maintainer='Robert Wilson',
      author_email='rwi@pml.ac.uk',

      packages = ["ncplot"],
      setup_requires=[
        'setuptools',
        'numpy',
        'setuptools-git',
        'wheel',
    ],
      install_requires = REQUIREMENTS,
      extras_require = extras_require,
      zip_safe=False)




