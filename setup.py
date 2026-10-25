from setuptools import setup, find_packages

setup(
    name="tisae",
    version="1.0.0",
    description="Temporal Island Sparse Autoencoders for Clinical Time Series",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4,<2.7",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "baselines": [
            "captum>=0.6.0",
            "dynamask",
            "timeshap",
            "shap<0.43",
            "numpy<2.0",
            "pytorch-lightning>=2.0",
        ],
        "llm": [
            "anthropic>=0.25",
            "openai>=1.0",
        ],
    },
)
