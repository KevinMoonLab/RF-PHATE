import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rfphate',
    version='0.0.1',
    author='Jake Rhodes',
    author_email='jake.rhodes@usu.edu',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jakerhodes/rfphate',
    project_urls = {
        "Bug Tracker": "https://github.com/jakerhodes/rfphate/issues"
    },
    license='GNU-3',
    packages=['rfphate'],
<<<<<<< HEAD
    install_requires=['sklearn', 'phate', 'pandas', 'seaborn'],
)
=======
    install_requires=['sklearn', 'phate', 'pandas', 'seaborn'],
)
>>>>>>> a0a9276fa2b045b5613149eeacd61c49fdcb3022
