from setuptools import setup

setup(
    name="ALRA",
    description="Adaptively-thresholded Low Rank Approximation.",
    version="0.0.1",
    license="BSD-3-Clause",
    author="Pavlin Poličar",
    author_email="pavlin.g.p@gmail.com",
    install_requires=["numpy>=1.15.4", "scipy>=1.1.0", "fbpca>=1.0"],
)
