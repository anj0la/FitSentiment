from setuptools import setup, find_packages

# read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="FitSentiment",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)