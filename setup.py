"""
Build and install the package.
"""
from __future__ import division, print_function, absolute_import

from setuptools import setup, find_packages

NAME = 'crawlab_toolbox'
FULLNAME = NAME
AUTHOR = "Daniel Newman"
AUTHOR_EMAIL = 'danielnewman09@gmail.com'
LICENSE = "GNU GPLv3.0"
URL = "https://github.com/danielnewman09/crawlab_toolbox"
DOWNLOAD_URL = "https://github.com/danielnewman09/crawlab_toolbox/releases"
DESCRIPTION = "This package contains the functions necessary to create presentation-ready CRAWLAB plots"
KEYWORDS = 'CRAWLAB'
LONG_DESCRIPTION = DESCRIPTION

VERSION = '0.0.1'

PACKAGES = find_packages()
SCRIPTS = []

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "License :: OSI Approved :: {}".format(LICENSE),
]
PLATFORMS = "Any"
INSTALL_REQUIRES = ['matplotlib','numpy','scipy']

import sys

'''
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('crawlab-toolbox',parent_package,top_path)
    config.add_subpackage('.plotting')
    config.add_subpackage('.utilities')
    config.add_subpackage('.inputshaping')
    config.make_config_py()
    return config
'''

INSTALL_REQUIRES = ['matplotlib','numpy','scipy']

if __name__ == '__main__':
    
    metadata = dict(
            name=NAME,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            #cmdclass=cmdclass,
            classifiers=CLASSIFIERS,
            platforms=PLATFORMS,
            #setup_requires=INSTALL_REQUIRES,
            install_requires=INSTALL_REQUIRES,
            #python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    )
    setup(**metadata)

    #from numpy.distutils.core import setup
    #setup(**configuration(top_path='').todict())
