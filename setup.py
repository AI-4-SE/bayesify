import setuptools

setuptools.setup(
    name="bayesify",  # Replace with your own username
    version="0.1",
    python_requires=">=3.9, <3.10",
    author="Johannes Dorn",
    author_email="johannes.dorn@uni-leipzig.de",
    description="Uncertainty-aware NFP Predictions with Probabilistic Programming",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy~=1.22.3',
        'matplotlib~=3.5.1',
        'pandas~=1.4.2',
        'arviz~=0.12.0',
        'seaborn~=0.11.2',
        'scipy~=1.8.0',
        'scikit-learn~=1.1.0rc1',
        'scipy~=1.8.0',
        'numpyro[cpu]~=0.9.2',
        'pyro-ppl~=1.8.1',
        'networkx~=2.8',
        'statsmodels~=0.13.2'
    ],
)
