import io
from setuptools import setup, find_packages

setup(
    name='SWonQC',
    author='Bruno Senjean',
    author_email='bruno.senjean@umontpellier.fr',
    url='',
    description=('Schrieffer-Wolff on Quantum Computers'),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
)
