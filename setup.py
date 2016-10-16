from distutils.core import setup



setup(
    name='swanljung',
    version='0.1',
    packages=['swanljung.cubetools', 'swanljung.cubetools.biggusext',
              'swanljung.plotblocks', 'swanljung.jobtools'],
    url='https://github.com/jswanljung/swanljung',
    license='',
    author='Johan Swanljung',
    author_email='johan.swanljung@gmail.com',
    description='Personal python tools',
    package_dir = {'':'lib'}
)
