from setuptools import setup, find_packages

setup(
    name="houseprice",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "matplotlib>=3.8.4",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "scikit-learn>=1.4.2",
        "xgboost>=2.0.3",
        "seaborn>=0.13.0",
    ],
)
