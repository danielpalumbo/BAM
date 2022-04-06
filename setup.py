from setuptools import setup

setup(name='bam',
      version='1.0',
      description='Model fitting accretion flows around black holes.',
      url='http://github.com/danielpalumbo/BAM/',
      author='danielpalumbo',
      author_email='daniel.palumbo@cfa.harvard.edu',
      license='GPLv3',
      packages=['bam',
                'bam.inference'],
      install_requires=['numpy','scipy>=1.8','dynesty','matplotlib','ehtim','dill','scikit-image'])
