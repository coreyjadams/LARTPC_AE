from . import larcv_io



# Here, we set up a bunch of template IO formats in the form of callable functions:

# These are all doing sparse IO, so there is no dense IO template here.  But you could add it.

def train_io(input_file, max_voxels_2d, max_voxels_3d, producer_2d, producer_3d, prepend_names=""):
    
        
    proc_2d =  gen_sparse2d_data_filler(name=prepend_names + "data2d", producer=producer_2d, max_voxels=max_voxels_2d)
    proc_3d =  gen_sparse3d_data_filler(name=prepend_names + "data3d", producer=producer_3d, max_voxels=max_voxels_3d)


    config = larcv_io.ThreadIOConfig(name="TrainIO")

    config.add_process(proc_2d)
    config.add_process(proc_3d)

    config.set_param("InputFiles", input_file)

    return config


def test_io(input_file, max_voxels_2d, max_voxels_3d, data_producer, label_producer, prepend_names="aux_"):
    proc_2d =  gen_sparse2d_data_filler(name=prepend_names + "data2d", producer=producer_2d, max_voxels=max_voxels_2d)
    proc_3d =  gen_sparse3d_data_filler(name=prepend_names + "data3d", producer=producer_3d, max_voxels=max_voxels_3d)


    config = larcv_io.ThreadIOConfig(name="TestIO")

    config.add_process(proc_2d)
    config.add_process(proc_3d)

    config.set_param("InputFiles", input_file)

    return config


def ana_io(input_file, max_voxels_2d, max_voxels_3d, data_producer, label_producer, prepend_names=""):
    proc_2d =  gen_sparse2d_data_filler(name=prepend_names + "data2d", producer=producer_2d, max_voxels=max_voxels_2d)
    proc_3d =  gen_sparse3d_data_filler(name=prepend_names + "data3d", producer=producer_3d, max_voxels=max_voxels_3d)


    config = larcv_io.ThreadIOConfig(name="AnaIO")

    config.add_process(proc_2d)
    config.add_process(proc_3d)

    config.set_param("InputFiles", input_file)

    return config


def gen_sparse2d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor2D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("Tensor2DProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("Channels",          "[0,1,2]")
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "false")

    return proc

def gen_sparse3d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor3D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("Tensor3DProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "false")

    return proc


