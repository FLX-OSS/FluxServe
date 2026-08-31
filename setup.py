# MIT License

# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import io
import os
import re

from datetime import datetime
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('python', 'fluxserve', '__init__.py')

if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')

extensions = []
cmdclass = {}

setup(
    # Metadata
    name='fluxserve',
    version=VERSION,
    python_requires='>=3.9',
    description='A High-Performance Inference Serving Library for Diffusion Language Model',
    long_description_content_type='text/markdown',
    license='MIT License',

    # Package info
    packages=find_packages(where="python", exclude=(
        'tests',
    )),
    package_dir={"": "python"},
    package_data={'': [os.path.join('datasets', 'dataset_checksums', '*.txt')]},
    entry_points={
        "console_scripts": [
            "fluxserve=fluxserve.cli:main",
        ],
    },
    zip_safe=True,
    include_package_data=True,
    ext_modules=extensions,
    cmdclass=cmdclass,
)
