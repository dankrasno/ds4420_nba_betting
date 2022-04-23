from setuptools import find_packages, setup

setup(
    name="nba-betting",
    author="Dan Krasnonosenkikh, Aaditya Watwe",
    description="Unsupervised Ensemble Learning for NBA Betting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license=open("LICENSE").read(),
    packages=find_packages(include=("nba_betting", "nba_betting.*")),
    python_requires=">=3.8",
    install_requires=[
        "nba-api",
        "pandas",
        "numpy",
        "scikit-learn",
        "tabulate",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "mypy-extensions",
            "data-science-types",
            "pytest",
            "flake8",
            "flake8-print",
        ]
    },
    package_data={
        "nba_betting": ["py.typed"],
    },
)
