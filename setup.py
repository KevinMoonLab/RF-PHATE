import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='RUN-RF-PHATE',
    version='0.0.1',
    author='Jake Rhodes',
    author_email='jake.rhodes@usu.edu',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jakerhodes/RUN-RF-PHATE',
    project_urls = {
        "Bug Tracker": "https://github.com/jakerhodes/RUN-RF-PHATE/issues"
    },
    license='GNU-3',
    packages=[],
    install_requires=['sklearn', 'phate'],
)