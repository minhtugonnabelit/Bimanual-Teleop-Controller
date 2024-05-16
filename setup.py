from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["bimanual_controller","perception"],
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "spatialmath-python", "roboticstoolbox-python"],
)

setup(**d)