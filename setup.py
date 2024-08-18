from setuptools import setup

setup(
    name='nails',
    version='1.0.0',
    packages=['helper', 'helper.base', 'helper.utils', 'helper.losses', 'helper.metrics', 'helper.datasets',
              'helper.decoders', 'helper.decoders.fpn', 'helper.decoders.pan', 'helper.decoders.unet',
              'helper.decoders.manet', 'helper.decoders.pspnet', 'helper.decoders.linknet', 'helper.decoders.deeplabv3',
              'helper.decoders.unetplusplus', 'helper.encoders'],
    url='https://www.ictp.it/',
    license='GPL-3.0 license',
    author='ICTP',
    author_email='ltenze@gmail.com',
    description='We present an algorithm that allows to identify color variations from fingernails using Artificial Intelligence (A.I.). Healthy nails have a uniform color and are visually smooth. However, as one ages, nails may become more brittle and may have discoloration due to injury, fungal and viral infections, medications, etc. Furthermore, the appearance of the fingernails can also change due to some medical conditions.'
    install_requires = [
        'timm==0.4.12',
        'segmentation-models-pytorch==0.3.0',
        'matplotlib',
        'numpy',
        'opencv-python'
    ]
)
