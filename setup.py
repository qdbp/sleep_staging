from setuptools import setup

# semver with automatic minor bumps keyed to unix time
__version__ = "0.5.0"

setup(
    python_requires=">=3.6.0",
    install_requires=["numpy", "scipy"],
    extras_require={"test": ["pytest"]},
    name="sleep_staging",
    packages=["sleep_staging"],
    entry_points={
        "console_scripts": ["sleep_staging = sleep_staging.main:main"]
    },
    version=__version__,
)
