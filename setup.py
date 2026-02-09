from setuptools import setup, find_packages

setup(
    name="surpass-data-collection",
    version="0.1.0",
    description="SURPASS data collection utilities",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        # put runtime deps here, e.g.
        # "numpy",
        # "rospy",
    ],
)
