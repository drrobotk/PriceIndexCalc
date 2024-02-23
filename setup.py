from gettext import find
from setuptools import setup, find_packages

setup(
    name='PriceIndexCalc',
    version='0.7',
    description='Price Index Calculator using bilateral and multilateral methods',
    author='Dr. Usman Kayani',
    url='https://github.com/drrobotk/PriceIndexCalc',
    author_email='usman.kayaniphd@gmail.com',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[
        'pandas', 
        'numpy', 
        'scipy',
        'scikit-learn',
        'seaborn',
        'statsmodels',
        'pyspark'
        ],
    include_package_data=True,
)