import setuptools

# Use the text in the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup metadata for initializing the library
setuptools.setup(
    name="CureQ",
    version="1.2.15",
    author="CureQ",
    author_email="cureq-ft@hva.nl",
    description="Library for analyzing MEA files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CureQ/CureQ.git",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib>=3.7.3",        # For visualisation plots of the data
        "numpy>=1.26.4",            # For array computing in python
        "h5py>=3.9.0",              # For reading the HDF5 file in python
        "pandas>=2.1.4",            # For creating and using a 2D Dataframe
        "scipy>=1.11.4",            # For scientific functions in Python
        "seaborn>=0.12.2",          # For statistical data visualization
        "statsmodels>=0.14.0",      # For calculating the partial autocorrelation function
        "scikit-image>=0.22.0",     # For determining the threshold of the network bursts
	    "KDEpy>=1.1.9",		        # For fast and efficient Kernel Density Estimation in Python
        "customtkinter>=5.2.2",     # Graphical user interface
        "CTkToolTip>=0.8",          # customtkinter tooltip
        "CTkMessagebox>=2.7",       # ctk messagebox widget
        "CTkColorPicker>=0.9.0",    # ctk colorpicker widget
        'requests>=2.32.3',         # Used to get information about newest available version of package
        'pyshortcuts>=1.9.5'        # Create desktop shortcuts
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_data={
        "CureQ":['GUI/MEAlytics_logo.ico', 'GUI/theme.json']
    },
    include_package_data=True,
    py_modules=["main"],
    entry_points={
        'console_scripts': [
            'cureq=CureQ.__main__:main',
        ],
    },
)
