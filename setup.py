import codecs
import os
import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


with open('README.md', 'r') as fh:
    description = fh.read()

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + './requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()


setuptools.setup(
    name='nustar_tools',
    author='Reed B. Masek',
    author_email='masek014@umn.edu',
    include_package_data=True,
    packages=['nustar_tools', 'nustar_tools.mapping', 'nustar_tools.pixels', 'nustar_tools.plotting', 'nustar_tools.regions', 'nustar_tools.trackers', 'nustar_tools.utils'],
    description='General NuSTAR tools for plotting, mapping, and data manipulation implemented in Python.',
    version=get_version('nustar_tools/__init__.py'),
    long_description=description,
    long_description_content_type='text/markdown',
    url='https://github.umn.edu/MASEK014/nustar_tools',
    license='MIT',
    python_requires='>=3.10',
    install_requires=install_requires
)