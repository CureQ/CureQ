import setuptools

# Use the text in the README file for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Setup metadata for initializing the library
setuptools.setup(
    name="CureQ",
    version="0.0.2",
    author="Jesse Antonissen",
    author_email="cureq-ft@hva.nl",
    description="Library for analyzing MEA files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CureQ/CureQ.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)