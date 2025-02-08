import setuptools

# Use the text in the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup metadata for initializing the library
setuptools.setup(
    name="CureQ",
    version="1.2.9",
    author="CureQ",
    author_email="cureq-ft@hva.nl",
    description="Library for analyzing MEA files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CureQ/CureQ.git",
    packages=setuptools.find_packages(),
    install_requires=[          # Installs all required libraries to run the pipeline analysis correctly
        "matplotlib>=3.7.3",    # For visualisation plots of the data
        "numpy>=1.26.4",        # For array computing in python
        "h5py>=3.9.0",          # For reading the HDF5 file in python
        "pandas>=2.1.4",        # For creating and using a 2D Dataframe
        "scipy>=1.11.4",        # For scientific functions in Python
        "scikit-learn>=1.3.0",  # For basic Machine Learning modules
        "seaborn>=0.12.2",      # For statistical data visualization
        "statsmodels>=0.14.0",  # For calculating the partial autocorrelation function
        "scikit-image>=0.22.0", # For determining the threshold of the network bursts
        "plotly>=5.14.0",       # For creating an interactive 3D view of a single well
	    "KDEpy>=1.1.9",		    # For Kernel Density Estimation in Python
        "customtkinter>=5.2.2", # Graphical user interface
        "CTkToolTip>=0.8",      # customtkinter tooltip
        "CTkMessagebox>=2.7",   # ctk messagebox widget
        "CTkColorPicker>=0.9.0",# ctk colorpicker widget
        "winshell>=0.6",
        "pywin32>=308",
        'requests>=2.32.3'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_data={
        "CureQ":['cureq_icon.ico', 'theme.json']
    },
    include_package_data=True,
    py_modules=["main"],
    entry_points={
        'console_scripts': [
            'cureq=CureQ.main:main',
        ],
    },
)
