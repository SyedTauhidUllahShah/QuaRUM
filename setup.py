"""
QuaRUM Package Setup Configuration

This file defines the metadata and dependencies for the 'quarum' package,
which provides a CLI tool for generating UML domain models from requirements
documents using qualitative data analysis and retrieval augmented methods.

Key Components:
- CLI entry point: 'quarum=quarum.cli:main'
- Core dependencies: LangChain, OpenAI, FAISS, NumPy
- Target audience: Developers and researchers in software requirements analysis
- License: MIT License

The package enables automated domain model creation by processing textual
requirements and leveraging AI-powered data retrieval techniques.
"""

from setuptools import setup, find_packages

setup(
    name="quarum",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain_openai>=0.0.2",
        "langchain_community>=0.0.1",
        "langchain_text_splitters>=0.0.1",
        "langchain_core>=0.1.0",
        "openai>=1.3.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "quarum=quarum.cli:main",
        ],
    },
    author="Syed Tauhid Ullah Shah",
    author_email="Syed.tauhidullahshah@ucalgary.ca",
    description="QuaRUM: Qualitative Data Analysis-Based Retrieval Augmented UML Domain Model from Requirements Documents",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SyedTauhidUllahShah/QuaRUM",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
