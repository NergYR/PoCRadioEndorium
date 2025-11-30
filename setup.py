from setuptools import setup, find_packages

setup(
    name="radio-airsoft-simulator",
    version="0.1.0",
    description="Simulation de système radio pour airsoft avec étalement de spectre",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "cryptography>=41.0.0",
        "pycryptodome>=3.19.0",
    ],
    python_requires=">=3.8",
)
