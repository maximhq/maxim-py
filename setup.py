import codecs
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "3.7.2"
DESCRIPTION = "Maxim Python Library"
LONG_DESCRIPTION = "A package that allows you to use the Maxim Python Library to interact with the Maxim Platform"

print("=============================")
print("Releasing version: ", VERSION)
print("=============================")

packages = find_packages()
packages.remove("maxim.tests")
# Setting up
setup(
    name="maxim-py",
    version=VERSION,
    author="Maxim Engineering",
    author_email="<eng@getmaxim.ai>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=packages,
    install_requires=["requests", "urllib3", "typing-extensions", "filetype"],
    extras_require={},
    keywords=["python", "prompts", "logs", "workflow", "testing"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)