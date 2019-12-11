from larcv import larcv
import numpy




def count_2d(file_name, product, producer):
    io = larcv.IOManager()
    io.add_in_file(file_name)
    io.initialize()
    voxel_counts = numpy.zeros((io.get_n_entries(), 3))

    for i in range(io.get_n_entries()):
        io.read_entry(i)
        image = larcv.EventSparseTensor2D.to_sparse_tensor(io.get_data("sparse2d", producer))
        for plane in [0,1,2]:
            voxel_counts[i][plane] = image.as_vector()[plane].size()
            meta = image.as_vector()[plane].meta()
        # image3d = io.get_data("sparse3d", "sbndvoxels")
        # voxel_counts3d[i] = image3d.as_vector().size()

        if i % 1000 == 0:
            print("On entry ", i, " of ", io.get_n_entries())

        if i > 100:
            break

    print ("Average Voxel Occupation: ")
    for p in [0,1,2]:
        print("  {p}: {av:.2f} +/- {rms:.2f} ({max} max)".format(
            p   = p, 
            av  = numpy.mean(voxel_counts[:,p]), 
            rms = numpy.std(voxel_counts[:,p]), 
            max = numpy.max(voxel_counts[:,p])
            )
        )
    print("Image shapes in dense representation are: ")
    for p in [0,1,2]:
        print(image.as_vector()[plane].meta().dump())


def count_3d(file_name, product, producer):
    io = larcv.IOManager()
    io.add_in_file(file_name)
    io.initialize()
    voxel_counts3d = numpy.zeros((io.get_n_entries(), 1))
    for i in range(io.get_n_entries()):
        io.read_entry(i)
        image3d = larcv.EventSparseTensor3D.to_sparse_tensor(io.get_data("sparse3d", producer))
        voxel_counts3d[i] = image3d.as_vector()[0].size()

        if i % 1000 == 0:
            print("On entry ", i, " of ", io.get_n_entries())

        if i > 100:
            break
    print(" 3D: {av:.2f} +/- {rms:.2f} ({max} max)".format(
        av  = numpy.mean(voxel_counts3d[:]), 
        rms = numpy.std(voxel_counts3d[:]), 
        max = numpy.max(voxel_counts3d[:])
        )
    )

    print("Image shapes in dense representation are: ")
    print(image3d.as_vector()[0].meta().dump())



if __name__ == '__main__':
    count_2d("/Users/corey.adams/data/DUNE/pixsim_small/test.h5", "sparse2d", "dunevoxels")
    count_3d("/Users/corey.adams/data/DUNE/pixsim_small/test.h5", "sparse3d", "dunevoxels")


