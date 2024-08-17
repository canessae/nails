from setuptools import setup

setup(
    name='nails',
    version='1.0.0',
    packages=['helper', 'helper.base', 'helper.utils', 'helper.losses', 'helper.metrics', 'helper.datasets',
              'helper.decoders', 'helper.decoders.fpn', 'helper.decoders.pan', 'helper.decoders.unet',
              'helper.decoders.manet', 'helper.decoders.pspnet', 'helper.decoders.linknet', 'helper.decoders.deeplabv3',
              'helper.decoders.unetplusplus', 'helper.encoders'],
    url='https://www.ictp.it/',
    license='',
    author='ICTP',
    author_email='ltenze@gmail.com',
    description=''
    install_requires = [
        'timm==0.4.12',
        'segmentation-models-pytorch==0.3.0',
        'matplotlib',
        'numpy',
        'opencv-python'
    ]
)
