from setuptools import setup, find_packages

setup(
    name="stardist2d",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=PYTHON_VERSION",
    install_requires=[
    ],
)
