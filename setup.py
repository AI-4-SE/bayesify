import setuptools

setuptools.setup(
    name="bayesify",  # Replace with your own username
    version="0.1.1",
    python_requires=">=3.7, <3.11",
    author="Johannes Dorn",
    author_email="johannes.dorn@uni-leipzig.de",
    description="Uncertainty-aware NFP Predictions with Probabilistic Programming",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy~=1.23",
        "matplotlib~=3.5.1",
        "pandas~=1.5",
        "arviz==0.12.1",
        "seaborn~=0.11.2",
        "scipy~=1.8.0",
        "scikit-learn~=1.1.0rc1",
        "numpyro[cpu]~=0.10",
        "networkx~=2.8",
        "statsmodels~=0.13.2",
    ],
)
