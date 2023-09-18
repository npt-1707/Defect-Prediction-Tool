from setuptools import setup, find_packages

VERSION = '0.1.8'
DESCRIPTION = 'A cutting-edge defect prediction tool with up-to-date Just-in-Time techniques and a robust API, empowering software development teams to proactively identify and mitigate defects in real-time'

# Setting up
setup(
    name="defectguard",
    version=VERSION,
    author="NeuralNine (Florian Dedov)",
    author_email="<mail@neuralnine.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    package_data={
        "defectguard.utils": ["*.json"],
    },
    install_requires=['numpy', 'pandas', 'PyGithub', 'Requests', 'tqdm', 'scikit-learn', 'torch', 'imblearn', 'scipy', 'dvc', 'dvc-gdrive'],
    keywords=['python', 'defect', 'prediction', 'just-in-time', 'defect prediction'],
    entry_points={
        'console_scripts': ['defectguard=defectguard:main'],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)