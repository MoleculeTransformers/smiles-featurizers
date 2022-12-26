from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

required = []
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="smiles-featurizers",
    version="1.0.0",
    author="Shahrukh Khan",
    author_email="sk28671@gmail.com",
    description="A python library for extracting molecular SMILES embeddings from language models pre-trained with various objectives and/or architectures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MoleculeTransformers/smiles-featurizers",
    packages=[
        "smiles_featurizers"
    ],
    install_requires=required,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
