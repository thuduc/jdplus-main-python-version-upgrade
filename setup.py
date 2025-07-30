"""Setup script for JDemetra+ Python."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jdemetra-py",
    version="0.1.0",
    author="JDemetra+ Team",
    author_email="",
    description="Python implementation of JDemetra+ seasonal adjustment framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jdemetra/jdplus-python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy>=2.0,<3",
        "pandas>=2.2,<3",
        "scipy>=1.12,<2",
        "statsmodels>=0.14,<1",
        "matplotlib>=3.8,<4",
        "seaborn>=0.13,<1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "performance": [
            "numba>=0.59,<1",
        ],
    },
)