import numpy

'''
This is a torch-free file that exists to massage data
From sparse to dense or dense to sparse, etc.

This can also convert from sparse to sparse to rearrange formats
For example, larcv BatchFillerSparseTensor2D (and 3D) output data
with the format of
    [B, N_planes, Max_voxels, N_features]

where N_features is 2 or 3 depending on whether or not values are included
(or 3 or 4 in the 3D case)

# The input of a pointnet type format can work with this, but SparseConvNet
# requires a tuple of (coords, features, [batch_size, optional])


'''

from . import flags
FLAGS = flags.FLAGS()


def larcvsparse_to_dense_2d(input_array, dense_shape):

    batch_size = input_array.shape[0]
    n_planes   = input_array.shape[1]

    ###################################################################
    # We do something unexpected here.  The X direction in 2D
    # is swapped with Y, to align the dimensions better with 3D data
    ###################################################################

    if FLAGS.DATA_FORMAT == "channels_first":
        output_array = numpy.zeros((batch_size, n_planes, dense_shape[1], dense_shape[0]), dtype=numpy.float32)
    else:
        output_array = numpy.zeros((batch_size, dense_shape[1], dense_shape[0], n_planes), dtype=numpy.float32)


    x_coords = input_array[:,:,:,1]
    y_coords = input_array[:,:,:,0]
    val_coords = input_array[:,:,:,2]



    filled_locs = val_coords != -999
    non_zero_locs = val_coords != 0.0
    mask = numpy.logical_and(filled_locs,non_zero_locs)
    # Find the non_zero indexes of the input:
    batch_index, plane_index, voxel_index = numpy.where(filled_locs)


    values  = val_coords[batch_index, plane_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, plane_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, plane_index, voxel_index])

    # print(numpy.min(x_index))
    # print(numpy.min(y_index))
    # print()
    # print(numpy.max(x_index))
    # print(numpy.max(y_index))

    # Tensorflow expects format as either [batch, height, width, channel]
    # or [batch, channel, height, width]
    # Fill in the output tensor




    if FLAGS.DATA_FORMAT == "channels_first":
        output_array[batch_index, plane_index, x_index, y_index] = values
    else:
        output_array[batch_index, x_index, y_index, plane_index] = values


    return output_array



def larcvsparse_to_dense_3d(input_array, dense_shape, padding=[60,12,30]):

    # padding will increase the output size, and center the pixels into the output

    batch_size = input_array.shape[0]

    this_dense_shape = [d for d in dense_shape]

    print(this_dense_shape)
    for i in range(len(this_dense_shape)):
        this_dense_shape[i] += padding[i]

    offset = [p / 2 for p in padding ]

    if FLAGS.DATA_FORMAT == "channels_first":
        # output_array = numpy.zeros((batch_size, n_planes, this_dense_shape[0], this_dense_shape[1]), dtype=numpy.float32)
        output_array = numpy.zeros((batch_size,1) + tuple(this_dense_shape), dtype=numpy.float32)
    else:
        # output_array = numpy.zeros((batch_size, this_dense_shape[0], this_dense_shape[1], n_planes), dtype=numpy.float32)
        output_array = numpy.zeros((batch_size,) + tuple(this_dense_shape) +(1,), dtype=numpy.float32)


    # By default, this returns channels_first format with just one channel.
    # You can just reshape since it's an empty dimension, effectively

    x_coords   = input_array[:,:,0]
    y_coords   = input_array[:,:,1]
    z_coords   = input_array[:,:,2]
    val_coords = input_array[:,:,3]


    # Find the non_zero indexes of the input:
    batch_index, voxel_index = numpy.where(val_coords != -999)

    values  = val_coords[batch_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, voxel_index] + offset[0])
    y_index = numpy.int32(y_coords[batch_index, voxel_index] + offset[1])
    z_index = numpy.int32(z_coords[batch_index, voxel_index] + offset[2])


    # Fill in the output tensor
    if FLAGS.DATA_FORMAT == "channels_first":
        # output_array[batch_index, plane_index, y_index, x_index] = values
        output_array[batch_index, 0, x_index, y_index, z_index] = values
    else:
        output_array[batch_index, x_index, y_index, z_index, 0]  = values


    return output_array
