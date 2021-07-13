from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
        "sklearn",
        "opacus"
]

setup(
    name = 'FSL_algorithm',
    version = '1.0.0',
    install_requires=REQUIRED_PACKAGES,
    packages = find_packages(),
    include_package_data = True
)