from setuptools import setup, find_packages

setup(
    name="Datapresso",
    version="0.1.0",
    description="Datapresso Data Construction Framework for efficient model fine-tuning",
    author="Datapresso Team",
    author_email="team@datapresso.org",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "hydra-core>=1.3.0",
        "jsonlines>=3.0.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.0.0",
        "datasets>=2.10.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
