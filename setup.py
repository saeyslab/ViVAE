from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Single-cell dimensionality reduction toolbox'
LONG_DESCRIPTION = 'Deep learning-based single-cell dimensionality reduction framework with enhanced diagnostics tools'

setup(
        name='ViVAE', 
        version=VERSION,
        author="David Novak",
        author_email="<davidnovak9000@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python'],
        classifiers= [
            "Intended Audience :: Bioinformatics",
            "Programming Language :: Python :: 3"
        ]
)