from setuptools import setup

setup(name='dmc3d',
      version='1.0',
      description='Model fitting around black holes.',
      url='http://github.com/danielpalumbo/3DMC/',
      author='danielpalumbo',
      author_email='daniel.palumbo@cfa.harvard.edu',
      license='GPLv3',
      packages=['dmc3d',
                'dmc3d.inference'],
      install_requires=['numpy','scipy','theano','pymc3','matplotlib','ehtim'])
