# Autoencoders for LarTPCs

This network is designed to learn the cross plane correlations from 2D detectors and use them to promote 2D data into 3D data.  It does this with autoencoders: There is a pair of auto encoders, for 2D and for 3D, that share a common latent space.  By training them together and adding to the loss term a MSE between the latent vectors, we can guide the networks to the same internal representation after 2D and 3D encoding.  This allows an inference pass of the 2D encoder and the 3D decoder, effectively leveraging the autoencoders to solve the dimensional translation problem.

## Dataset

This network is meant to use the DUNE pixsim dataset from OSF, https://osf.io/kbn5a/.  This is a collection of approximately 10M events with both 2D and 3D projections in a DUNE-like geometry (no wrapped wires).

The native resolution of the images is:
 - [1536, 1024] times 3 planes in x, y in 2D (y here is the drift direction)
 - [900, 500, 1250] in x, y, z in 3D (x here is the drift direction).
 
The images do not map nicely on to each other in dimensionality.  For this reason, during training the 3D image is padded to a shape of [960, 512, 1280] (all divisible by 2**6) to enable simpler downsampling and upsampling.  The 2D images are swapped in dimension: they become shape [1024, 3planes, 1536] to make a nicer correspondance to the ordinality of the 3D images.

The dataformat is in larcv3 format, which enables sparseIO at scale with MPI-IO and hdf5.

# Requirements

This repository requires:
 - python 3.6+ 
 - pytorch
 - numpy
 - larcv3
 - tensorboardX
 - (horovod, MPI, mpi4py, larcv3+MPI for distributed training)

All of these are available via pip or github.

## Training

The training loop is initiated with the following command:
```
$ python ./bin/exec.py train --file /path/to/file/train.h5  -i 5 -df channels_first
```

So far, this network is early in development and the training process is not yet solidified.
