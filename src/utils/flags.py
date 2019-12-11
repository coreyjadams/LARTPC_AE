import argparse

import os,sys
top_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(top_dir)

'''
This script is heavily inspired by the following code from drinkingkazu:
https://github.com/DeepLearnPhysics/dynamic-gcnn/blob/develop/dgcnn/flags.py
'''

# This class is from here:
# http://www.aleax.it/Python/5ep.html
# Which is an incredibly simply and elegenant way
# To enforce singleton behavior
class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

# This function is to parse strings from argparse into bool
def str2bool(v):
    '''Convert string to boolean value

    This function is from stackoverflow:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Arguments:
        v {str} -- [description]

    Returns:
        bool -- [description]

    Raises:
        argparse -- [description]
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class FLAGS(Borg):
    '''This class implements global flags through static variables
    The static-ness is enforced by inheriting from Borg, which it calls first and
    foremost in the constructor.

    All classes derived from FLAGS should call this constructor first

    It also has the ability to parse the arguments pass on the command line for overrides,
     and can print out help for all of the flag variables

    In particular, this base class defines a lot of 'shared' parameter configurations:
    IO, snapshotting directories, etc.

    Network parameters are not implemented here but in derived classes.
    '''

    def __init__(self):
        Borg.__init__(self)


    def _set_defaults(self):
        # Parameters controlling training situations
        self.COMPUTE_MODE          = "GPU"
        self.TRAINING              = True
        self.MINIBATCH_SIZE        = 2
        self.CHECKPOINT_ITERATION  = 100
        self.SUMMARY_ITERATION     = 1
        self.LOGGING_ITERATION     = 1
        self.LEARNING_RATE         = 0.0003
        self.ITERATIONS            = 5000
        self.VERBOSITY             = 0
        self.LOG_DIRECTORY         = './log'
        self.CHECKPOINT_DIRECTORY  = None

        self.DISTRIBUTED           = False

        # To be clear, this is specifying the image mode from larcv ThreadIO,
        # Not the input to the network

        # IO parameters
        # IO has a 'default' file configuration and an optional
        # 'auxilliary' configuration.  In Train mode, the default
        # is the training data, aux is testing data.
        # In inference mode, default is the validation data,
        # aux is the outputdata
        self.FILE                  = None
        self.IO_VERBOSITY          = 3
        # For this classification task, the label can be split or all-in-one

        # These are "background" parameters
        # And are meant to be copied to the 'KEYWORD_LABEL' area


        # Optional Test IO parameters:
        # To activate the auxilliary IO, the AUX file must be not None
        self.AUX_FILE              = None
        self.AUX_IO_VERBOSITY      = 3
        self.AUX_MINIBATCH_SIZE    = self.MINIBATCH_SIZE
        self.AUX_ITERATION         = 10*self.SUMMARY_ITERATION


        self.OPTIMIZER             = "Adam"
        self.REGULARIZE_WEIGHTS    = 0.0001

        self.PRODUCER_2D           = "dunevoxels"
        self.PRODUCER_3D           = "dunevoxels"

        # In the raw DUNE pixsim files, the occupancy is:
        #   3D: 1920.00 +/- 2772.85 (35849.0 max)
        #   2D Average Voxel Occupation:
        #       0: 1382.42 +/- 1695.47 (17589.0 max)
        #       1: 1314.12 +/- 1594.97 (16840.0 max)
        #       2: 1486.68 +/- 1831.28 (19428.0 max)
        self.MAX_VOXELS_2D         = 20000
        self.MAX_VOXELS_3D         = 36000

        # Raw shapes of the data.
        # Note that the 3D sparse -> dense transform applies a padding to make a slightly nicer shape
        # this is in data_transforms.py
        self.RAW_SHAPE_2D          = [1536, 1024]
        self.RAW_SHAPE_3D          = [900, 500, 1250]

        self.LATENT_SPACE          = [15, 8, 20]


        self.DATA_FORMAT           = "channels_first"


    def _add_default_io_configuration(self, parser):

        # IO PARAMETERS FOR INPUT:
        parser.add_argument('-f','--file', type=str, default=self.FILE,
            help="IO Configuration File [default: {}]".format(self.FILE))
        parser.add_argument('--io-verbosity', type=int, default=self.IO_VERBOSITY,
            help="IO verbosity [default: {}]".format(self.IO_VERBOSITY))

        parser.add_argument('-mb','--minibatch-size',type=int, default=self.MINIBATCH_SIZE,
            help="Number of images in the minibatch size [default: {}]".format(self.MINIBATCH_SIZE))
        return parser


    def _add_aux_io_configuration(self, parser):

        # IO PARAMETERS FOR INPUT:
        parser.add_argument('--aux-file', type=str, default=self.AUX_FILE,
            help="IO Configuration File [default: {}]".format(self.AUX_FILE))
        parser.add_argument('--aux-io-verbosity', type=int, default=self.AUX_IO_VERBOSITY,
            help="IO verbosity [default: {}]".format(self.AUX_IO_VERBOSITY))

        parser.add_argument('--aux-iteration',type=int, default=self.AUX_ITERATION,
            help="Iteration to run the aux operations [default: {}]".format(self.AUX_ITERATION))

        parser.add_argument('--aux-minibatch-size',type=int, default=self.AUX_MINIBATCH_SIZE,
            help="Number of images in the minibatch size [default: {}]".format(self.AUX_MINIBATCH_SIZE))

        return parser

    def _create_parsers(self):

        self._parser = argparse.ArgumentParser(description="Configuration Flags")

        subparsers = self._parser.add_subparsers(title="Modules",
                                                 description="Valid subcommands",
                                                 dest='mode',
                                                 help="Available subcommands: train, iotest, inference")



        # train parser
        self.train_parser = subparsers.add_parser("train", help="Train")
        self.train_parser.add_argument('-lr','--learning-rate', type=float, default=self.LEARNING_RATE,
                                  help='Initial learning rate [default: {}]'.format(self.LEARNING_RATE))
        self.train_parser.add_argument('-si','--summary-iteration', type=int, default=self.SUMMARY_ITERATION,
                                  help='Period (in steps) to store summary in tensorboard log [default: {}]'.format(self.SUMMARY_ITERATION))
        self.train_parser.add_argument('-li','--logging-iteration', type=int, default=self.LOGGING_ITERATION,
                                  help='Period (in steps) to print values to log [default: {}]'.format(self.LOGGING_ITERATION))
        self.train_parser.add_argument('-ci','--checkpoint-iteration', type=int, default=self.CHECKPOINT_ITERATION,
                                  help='Period (in steps) to store snapshot of weights [default: {}]'.format(self.CHECKPOINT_ITERATION))

        self.train_parser.add_argument('-o', '--optimizer', default=self.OPTIMIZER, type=str,
            choices=['lars', 'rmsprop', 'adam'],
            help="Optimizer to use, must be lars, rmsprop, adam [default: {}]".format(self.OPTIMIZER))

        self.train_parser.add_argument('-rw','--regularize-weights', type=float, default=self.REGULARIZE_WEIGHTS,
            help="Regularization strength for all learned weights [default: {}]".format(self.REGULARIZE_WEIGHTS))

        # attach common parsers
        self.train_parser  = self._add_default_network_configuration(self.train_parser)
        self.train_parser  = self._add_default_io_configuration(self.train_parser)
        self.train_parser  = self._add_aux_io_configuration(self.train_parser)
        self.train_parser  = self._add_core_configuration(self.train_parser)




        # IO test parser
        self.iotest_parser = subparsers.add_parser("iotest", help="Test io only (no network)")
        self.iotest_parser  = self._add_default_network_configuration(self.iotest_parser)
        self.iotest_parser = self._add_default_io_configuration(self.iotest_parser)
        self.iotest_parser = self._add_aux_io_configuration(self.iotest_parser)
        self.iotest_parser  = self._add_core_configuration(self.iotest_parser)


        # # inference parser
        self.inference_parser = subparsers.add_parser("inference",help="Run inference (optional output)")
        self.inference_parser = self._add_default_network_configuration(self.inference_parser)
        self.inference_parser = self._add_default_io_configuration(self.inference_parser)
        self.inference_parser = self._add_aux_io_configuration(self.inference_parser)
        self.inference_parser = self._add_core_configuration(self.inference_parser)


    def _add_core_configuration(self, parser):
        # These are core parameters that are important for all modes:
        parser.add_argument('-i', '--iterations', type=int, default=self.ITERATIONS,
            help="Number of iterations to process [default: {}]".format(self.ITERATIONS))

        parser.add_argument('-d','--distributed', action='store_true', default=self.DISTRIBUTED,
            help="Run with the MPI compatible mode [default: {}]".format(self.DISTRIBUTED))
        parser.add_argument('-m','--compute-mode', type=str, choices=['CPU','GPU'], default=self.COMPUTE_MODE,
            help="Selection of compute device, CPU or GPU  [default: {}]".format(self.COMPUTE_MODE))
        parser.add_argument('-ld','--log-directory', default=self.LOG_DIRECTORY,
            help='Prefix (directory + file prefix) for tensorboard logs [default: {}]'.format(self.LOG_DIRECTORY))

        parser.add_argument('-cd','--checkpoint-directory', default=self.CHECKPOINT_DIRECTORY,
            help='Prefix (directory) for snapshots of weights [default: {}]'.format(self.CHECKPOINT_DIRECTORY))

        # parser.add_argument('--image-producer', default=self.IMAGE_PRODUCER,
            # help='Name of the image producer in the files [default: {}]'.format(self.IMAGE_PRODUCER))
        # parser.add_argument('--label-producer', default=self.LABEL_PRODUCER,
            # help='Name of the label producer in the files [default: {}]'.format(self.LABEL_PRODUCER))


        parser.add_argument('--max-voxels-2d', default=self.MAX_VOXELS_2D, type=int,
            help='Maximum number of voxels used in sparse IO [default: {}]'.format(self.MAX_VOXELS_2D))

        parser.add_argument('--max-voxels-3d', default=self.MAX_VOXELS_3D, type=int,
            help='Maximum number of voxels used in sparse IO [default: {}]'.format(self.MAX_VOXELS_3D))

        parser.add_argument('--verbosity', default=self.VERBOSITY, type=int,
            help='Verbosity of python calls [default: {}]'.format(self.VERBOSITY))

        parser.add_argument('-df','--data-format', type=str, default=self.DATA_FORMAT,
            help="Channels format in the tensor shape [default: {}]".format(self.DATA_FORMAT))

        return parser



    def parse_args(self):
        self._set_defaults()
        self._create_parsers()
        # mode = sys.argv[1]
        # try:
        #     hps_dict = json.loads(''.join(sys.argv[2:]))
        # except json.DecodeError:
        args = self._parser.parse_args()
        # else:
        #     # Can't do f-strings, this code runs python 2.7.
        #     # I am working on a 3.6 transisition ...
        #     args_str = ' '.join("--{key}={val}".format(key=key, val=val) for key,val in hps_dict.items())
        #     args = self._parser.parse_args(mode+' '+args_str)
        self.update(vars(args))

        if self.MODE == 'inference':
            self.TRAINING = False


    def dump_config(self):
        print(self.__str__())


    def get_config(str):
        return str.__str__()


    def __str__(self):
        try:
            _ = getattr(self, '_parser')
            s = "\n\n-- CONFIG --\n"
            for name in iter(sorted(vars(self))):
                if name != name.upper(): continue
                attribute = getattr(self,name)
                if type(attribute) == type(self._parser): continue
                # s += " %s = %r\n" % (name, getattr(self, name))
                substring = ' {message:{fill}{align}{width}}: {attr}\n'.format(
                       message=name,
                       attr = getattr(self, name),
                       fill='.',
                       align='<',
                       width=35,
                    )
                s += substring
            return s

        except AttributeError:
            return "ERROR: call parse_args()"


    def update(self, args):
        for name,value in args.items():
            if name in ['func']: continue
            setattr(self, name.upper(), args[name])
        # Take special care to reset the keyword label attribute
        # to match the label mode:

    def _add_default_network_configuration(self, parser):
        raise NotImplementedError("Must use a derived class which overrides this function")



class dimnet(FLAGS):

    def __init__(self):
        FLAGS.__init__(self)



    def _set_defaults(self):
        # Parameters to control the network implementation
        self.BATCH_NORM                             = True
        self.USE_BIAS                               = True
        self.RESIDUAL                               = False

        # These are parameters that are configured for encoders
        # and decoders in 2D and 3D.

        self.ENCODER_2D_N_INITIAL_FILTERS           = 8
        self.ENCODER_2D_BLOCKS_PER_LAYER            = 2
        self.ENCODER_2D_NETWORK_DEPTH               = 4


        self.ENCODER_3D_N_INITIAL_FILTERS           = 8
        self.ENCODER_3D_BLOCKS_PER_LAYER            = 2
        self.ENCODER_3D_NETWORK_DEPTH               = 4

        # self.DECODER_2D_N_INITIAL_FILTERS           = 6
        self.DECODER_2D_BLOCKS_PER_LAYER            = 2
        self.DECODER_2D_NETWORK_DEPTH               = 4

        # self.DECODER_3D_N_INITIAL_FILTERS           = 6
        self.DECODER_3D_BLOCKS_PER_LAYER            = 2
        self.DECODER_3D_NETWORK_DEPTH               = 4

        self.NPLANES                                = 3



        # Run with half precision:
        self.INPUT_HALF_PRECISION        = False
        self.MODEL_HALF_PRECISION        = False
        self.LOSS_SCALE                  = 1.0


        # Relevant parameters for running on KNL:
        self.INTER_OP_PARALLELISM_THREADS    = 2
        self.INTRA_OP_PARALLELISM_THREADS    = 128

        FLAGS._set_defaults(self)

    def _add_default_network_configuration(self, parser):

        parser.add_argument('-ub','--use-bias', type=str2bool, default=self.USE_BIAS,
            help="Whether or not to include bias terms in all mlp layers [default: {}]".format(self.USE_BIAS))
        parser.add_argument('-bn','--batch-norm', type=str2bool, default=self.BATCH_NORM,
            help="Whether or not to use batch normalization in all mlp layers [default: {}]".format(self.BATCH_NORM))
        parser.add_argument('--residual', type=str2bool, default=self.RESIDUAL,
            help="Use residual units instead of convolutions [default: {}]".format(self.RESIDUAL))

        # parser.add_argument('--n-initial-filters', type=int, default=self.N_INITIAL_FILTERS,
        #     help="Number of filters applied, per plane, for the initial convolution [default: {}]".format(self.N_INITIAL_FILTERS))
        # parser.add_argument('--res-blocks-per-layer', type=int, default=self.RES_BLOCKS_PER_LAYER,
        #     help="Number of residual blocks per layer [default: {}]".format(self.RES_BLOCKS_PER_LAYER))
        # parser.add_argument('--network-depth', type=int, default=self.NETWORK_DEPTH,
        #     help="Total number of downsamples to apply [default: {}]".format(self.NETWORK_DEPTH))
        parser.add_argument('--nplanes', type=int, default=self.NPLANES,
            help="Number of planes to split the initial image into [default: {}]".format(self.NPLANES))

        parser.add_argument('--encoder-2d-n-initial-filters', type=int,
                            default=self.ENCODER_2D_N_INITIAL_FILTERS,
                            help="Number of initial filters in 2d encoder [default: {}]".format(
                                    self.ENCODER_2D_N_INITIAL_FILTERS)
                            )
        parser.add_argument('--encoder-2d-blocks-per-layer', type=int,
                            default=self.ENCODER_2D_BLOCKS_PER_LAYER,
                            help="Number of blocks per resolution layer in the 2D encoder [default: {}]".format(
                                    self.ENCODER_2D_BLOCKS_PER_LAYER)
                            )
        parser.add_argument('--encoder-2d-network-depth', type=int,
                            default=self.ENCODER_2D_NETWORK_DEPTH,
                            help="Number of resolution changes in the 2D encoder (downsampling) [default: {}]".format(
                                    self.ENCODER_2D_NETWORK_DEPTH)
                            )

        parser.add_argument('--encoder-3d-n-initial-filters', type=int,
                            default=self.ENCODER_3D_N_INITIAL_FILTERS,
                            help="Number of initial filters in 3d encoder [default: {}]".format(
                                self.ENCODER_3D_N_INITIAL_FILTERS)
                            )
        parser.add_argument('--encoder-3d-blocks-per-layer', type=int,
                            default=self.ENCODER_3D_BLOCKS_PER_LAYER,
                            help="Number of blocks per resolution layer in the 3D encoder [default: {}]".format(
                                self.ENCODER_3D_BLOCKS_PER_LAYER)
                            )
        parser.add_argument('--encoder-3d-network-depth', type=int,
                            default=self.ENCODER_3D_NETWORK_DEPTH,
                            help="Number of resolution changes in the 3D encoder (downsampling) [default: {}]".format(
                                self.ENCODER_3D_NETWORK_DEPTH)
                            )

        # parser.add_argument('--decoder_2d-n-initial-filters', type=int,
        #                     default=self.DECODER_2D_N_INITIAL_FILTERS,
        #                     help="Number of initial filters in 2d decoder [default: {}]".format(
        #                         self.DECODER_2D_N_INITIAL_FILTERS)
        #                     )
        parser.add_argument('--decoder-2d-blocks-per-layer', type=int,
                            default=self.DECODER_2D_BLOCKS_PER_LAYER,
                            help="Number of blocks per resolution layer in the 2D decoder [default: {}]".format(
                                self.DECODER_2D_BLOCKS_PER_LAYER)
                            )
        parser.add_argument('--decoder-2d-network-depth', type=int,
                            default=self.DECODER_2D_NETWORK_DEPTH,
                            help="Number of resolution changes in the 2D decoder (upsampling) [default: {}]".format(
                                self.DECODER_2D_NETWORK_DEPTH)
                            )

        # parser.add_argument('--decoder-3d-n-initial-filters', type=int,
        #                     default=self.DECODER_3D_N_INITIAL_FILTERS,
        #                     help="Number of initial filters in 3d decoder [default: {}]".format(
        #                         self.DECODER_3D_N_INITIAL_FILTERS)
        #                     )
        parser.add_argument('--decoder_3d-blocks-per-layer', type=int,
                            default=self.DECODER_3D_BLOCKS_PER_LAYER,
                            help="Number of blocks per resolution layer in the 3D decoder [default: {}]".format(
                                self.DECODER_3D_BLOCKS_PER_LAYER)
                            )
        parser.add_argument('--decoder-3d-network-depth', type=int,
                            default=self.DECODER_3D_NETWORK_DEPTH,
                            help="Number of resolution changes in the 3D decoder (upsampling) [default: {}]".format(
                                self.DECODER_3D_NETWORK_DEPTH)
                            )



        parser.add_argument('--model-half-precision', type=str2bool, default=self.MODEL_HALF_PRECISION,
            help="Use half precision for model weights and parameters [default: {}]".format(self.MODEL_HALF_PRECISION))

        parser.add_argument('--input-half-precision', type=str2bool, default=self.INPUT_HALF_PRECISION,
            help="Use half precision for input values and intermediate activations [default: {}]".format(self.INPUT_HALF_PRECISION))

        parser.add_argument('--loss-scale', type=float, default=self.LOSS_SCALE,
            help="Amount to scale the loss function before back prop [default: {}]".format(self.LOSS_SCALE))

        parser.add_argument('--inter-op-parallelism-threads',type=int, default=self.INTER_OP_PARALLELISM_THREADS,
            help="Passed to tf configproto [default: {}]".format(self.INTER_OP_PARALLELISM_THREADS))
        parser.add_argument('--intra-op-parallelism-threads',type=int, default=self.INTRA_OP_PARALLELISM_THREADS,
            help="Passed to tf configproto [default: {}]".format(self.INTRA_OP_PARALLELISM_THREADS))



        return parser
