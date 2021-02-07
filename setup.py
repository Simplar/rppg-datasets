import pathlib
from setuptools import setup, find_packages

DIR = pathlib.Path(__file__).parent
SRC = DIR / 'src'

requirements = (DIR / "requirements.txt").read_text().splitlines()
licence = (DIR / "LICENSE").read_text()
readme = (DIR / "README.md").read_text()


setup(
    name='rppg-datasets',
    version='0.0.1',
    packages=find_packages(where=str(SRC)),
    package_dir={'': str(SRC)},
    url='https://github.com/simplar/rppg_datasets',
    install_requires=requirements,
    license=license,
    keywords='rPPG',
    author='Konstantin Kalinin, Mikhail Kopeliovich',
    author_email='kopeliovich.mikhail@gmail.com',
    # package_data={'': [requirements, readme, licence]},
    description='Loaders of data from public rPPG datasets',
    long_description=readme
)
