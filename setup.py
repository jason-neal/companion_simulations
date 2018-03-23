"""Setup.py for Companion Simulations."""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# from setuptools import find_packages

config = {
    'description': 'Companion simulations of Crires spectra.',
    'author': 'Jason Neal',
    'url': 'https://github.com/jason-neal/companion_simulations.git',
    'download_url': 'https://github.com/jason-neal/companion_simulations.git',
    'author_email': 'jason.neal@astro.up.pt',
    'version': '0.1',
    'license': 'MIT',
    'setup_requires': ['pytest-runner'],
    'install_requires': ['pytest'],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    'extras_require': {
        'dev': ['check-manifest'],
        'test': ['coverage', 'pytest', 'pytest-cov', 'python-coveralls', 'hypothesis'],
    },
    'packages': ["mingle", "mingle/models", 'mingle/utilities', 'simulators'],
    # find_packages("src", exclude=['contrib', 'docs', 'tests']),

    'scripts': ["simulators/bhm_script.py",
                "simulators/iam_script.py",
                "simulators/tcm_script.py",
                "simulators/minimize_iam.py",
                "simulators/minimize_bhm.py",
                "bin/coadd_analysis_script.py",
                "bin/coadd_bhm_analysis.py",
                "bin/check_result.py",
                "bin/coadd_chi2_db.py",
                "bin/coadd_bhm_db.py",
                "bin/do_iam_on_stars.py",
                # "bin/fully_analyze_star.py",
                # "bin/create_min_chi2_table.py",
                "bin/iam_fake_full_stack.py",
                "bin/bhm_fake_full_stack.py",
                # "bin/analysis_iam_chi2.py"
                "bin/radius_ratio.py",
                ],
    'name': 'companion_simulations',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    "classifiers": [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
    ],
    # What does your project relate to?
    "keywords": ['Astronomy', 'Near-infrared spectroscopy', "M-dwarfs", "Phoenix-ACES", "Simulations"],
}

setup(**config)
