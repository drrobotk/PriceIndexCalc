from setuptools import setup

setup(
    name='PriceIndexCalc',
    version='0.1',
    description='Price Index Calculator using bilateral and multilateral methods',
    author='Dr. Usman Kayani',
    url='https://github.com/drrobotk/PriceIndexCalc',
    author_email='usman.kayani@ons.gov.uk',
    license='MIT',
    packages=['PriceIndexCalc'],
    install_requires=['pandas', 'numpy', 'scipy'],
    include_package_data=True,
)