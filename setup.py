from setuptools import setup

setup(
    name='multilateral_index_calc',
    version='0.1',
    description='Price Index Calculator',
    author='Dr. Usman Kayani',
    url='https://github.com/drrobotk/multilateral_index_calc',
    author_email='usman.kayani@ons.gov.uk',
    license='MIT',
    packages=['multilateral_index_calc'],
    install_requires=['pandas', 'numpy', 'scipy'],
    include_package_data=True,
)