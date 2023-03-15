# NuSTAR Tools

### Relevant Links

- The Github repository can be found [here](https://github.umn.edu/MASEK014/nustar_tools).
- The PyPI webpage can be found [here](https://test.pypi.org/project/nustar-tools/).

---

## Installation

The package is deployed on the Test PyPI, but it will eventually be moved to the real PyPI.

Install `nustar_tools` through pip:
> pip install -i https://test.pypi.org/simple/ nustar-tools

To build the documentation, `pdoc3` is required and the Git repository must be cloned.
Execute the following commands to build the documentation:
```
git clone https://github.umn.edu/MASEK014/nustar_tools.git
cd nustar_tools/
pdoc3 --html --output-dir docs
```
You will find the documentation in HTML files in the newly created `docs/` directory.
Begin with `index.html`.

### Requirements

`nustar_tools` was tested on Python 3.11.
It is currently unknown how it will perform on other versions of Python.

This package has several dependencies that are listed in `requirements.txt`.
These packages can be installed all at once by locally saving the `requirements.txt` file and executing the following command:
> pip install -r requirements.txt

The only package that must be installed manually is `nustar_pysolar`, which can be found [here](https://github.com/NuSTAR/nustar_pysolar).
While the installation instructions make use of a Conda environment, this is not necessary in order to install the package.
Follow the instructions as outlined on the page *except execute the same pip command from above instead of what's listed on the page when installing the requirements*.