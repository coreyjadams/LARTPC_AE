import os
import sys
import time
import tempfile
import itertools
from collections import OrderedDict

import numpy

import torch
# from torch.jit import trace

from larcv import queueloader


from . import flags
from . import data_transforms
from ..io import io_templates
from ..networks import encoders, decoders
FLAGS = flags.FLAGS()

import datetime

# This uses tensorboardX to save summaries and metrics to tensorboard compatible files.

import tensorboardX

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self,):
        if FLAGS.MODE == 'inference':
            mode = 'serial_access'
        else:
            mode = 'random_blocks'
        self._larcv_interface = queueloader.queue_interface(random_access_mode=mode)
        self._iteration       = 0
        self._global_step     = -1

        self._cleanup         = []

    def __del__(self):
        for f in self._cleanup:
            os.unlink(f.name)
    def _initialize_io(self):


        # Use the templates to generate a configuration string, which we store into a temporary file
        if FLAGS.TRAINING:
            config = io_templates.train_io(
                input_file=FLAGS.FILE,
                producer_2d=FLAGS.PRODUCER_2D,
                producer_3d=FLAGS.PRODUCER_3D,
                max_voxels_2d=FLAGS.MAX_VOXELS_2D,
                max_voxels_3d=FLAGS.MAX_VOXELS_3D)
        else:
            config = io_templates.ana_io(
                input_file=FLAGS.FILE,
                producer_2d=FLAGS.PRODUCER_2D,
                producer_3d=FLAGS.PRODUCER_3D,
                max_voxels_2d=FLAGS.MAX_VOXELS_2D,
                max_voxels_3d=FLAGS.MAX_VOXELS_3D)

        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        print(config.generate_config_str())

        main_file.close()
        self._cleanup.append(main_file)

        # Prepare data managers:
        io_config = {
            'filler_name' : config._name,
            'filler_cfg'  : main_file.name,
            'verbosity'   : FLAGS.VERBOSITY,
            'make_copy'   : True
        }

        data_keys = OrderedDict({
            'data2d': 'data2d',
            'data3d': 'data3d'
            })



        self._larcv_interface.prepare_manager('primary', io_config, FLAGS.MINIBATCH_SIZE, data_keys)

        # All of the additional tools are in case there is a test set up:
        if FLAGS.AUX_FILE is not None:

            if FLAGS.TRAINING:
                config = io_templates.test_io(
                    input_file=FLAGS.AUX_FILE,
                    producer_2d=FLAGS.PRODUCER_2D,
                    producer_3d=FLAGS.PRODUCER_3D,
                    max_voxels_2d=FLAGS.MAX_VOXELS_2D,
                    max_voxels_3d=FLAGS.MAX_VOXELS_3D)

                # Generate a named temp file:
                aux_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                aux_file.write(config.generate_config_str())

                aux_file.close()
                self._cleanup.append(aux_file)
                io_config = {
                    'filler_name' : config._name,
                    'filler_cfg'  : aux_file.name,
                    'verbosity'   : FLAGS.VERBOSITY,
                    'make_copy'   : True
                }



                self._larcv_interface.prepare_manager('aux', io_config, FLAGS.AUX_MINIBATCH_SIZE, data_keys)

            else:
                config = io_templates.ana_io(input_file=FLAGS.FILE, max_voxels=max_voxels)
                self._larcv_interface.prepare_writer(FLAGS.AUX_FILE)

    def init_network(self):


        self._encoder_2d = encoders.encoder2D(
                n_planes          = FLAGS.NPLANES,
                n_initial_filters = FLAGS.ENCODER_2D_N_INITIAL_FILTERS,
                batch_norm        = FLAGS.BATCH_NORM,
                use_bias          = FLAGS.USE_BIAS,
                residual          = FLAGS.RESIDUAL,
                depth             = FLAGS.ENCODER_2D_NETWORK_DEPTH,
                blocks_per_layer  = FLAGS.ENCODER_2D_BLOCKS_PER_LAYER,
                # latent_space      = FLAGS.LATENT_SPACE
            )
        self._encoder_3d = encoders.encoder3D(
                n_initial_filters = FLAGS.ENCODER_3D_N_INITIAL_FILTERS,
                batch_norm        = FLAGS.BATCH_NORM,
                use_bias          = FLAGS.USE_BIAS,
                residual          = FLAGS.RESIDUAL,
                depth             = FLAGS.ENCODER_3D_NETWORK_DEPTH,
                blocks_per_layer  = FLAGS.ENCODER_3D_BLOCKS_PER_LAYER,
                # latent_space      = FLAGS.LATENT_SPACE
            )

        self._decoder_2d = decoders.decoder2D(
                n_planes          = FLAGS.NPLANES,
                n_initial_filters = FLAGS.ENCODER_2D_N_INITIAL_FILTERS * 2**FLAGS.ENCODER_2D_NETWORK_DEPTH,
                batch_norm        = FLAGS.BATCH_NORM,
                use_bias          = FLAGS.USE_BIAS,
                residual          = FLAGS.RESIDUAL,
                depth             = FLAGS.DECODER_2D_NETWORK_DEPTH,
                blocks_per_layer  = FLAGS.DECODER_2D_BLOCKS_PER_LAYER,
                # latent_space      = FLAGS.LATENT_SPACE
            )
        self._decoder_3d = decoders.decoder3D(
                n_initial_filters = FLAGS.ENCODER_3D_N_INITIAL_FILTERS * 2**FLAGS.ENCODER_3D_NETWORK_DEPTH,
                batch_norm        = FLAGS.BATCH_NORM,
                use_bias          = FLAGS.USE_BIAS,
                residual          = FLAGS.RESIDUAL,
                depth             = FLAGS.DECODER_3D_NETWORK_DEPTH,
                blocks_per_layer  = FLAGS.DECODER_3D_BLOCKS_PER_LAYER,
                # latent_space      = FLAGS.LATENT_SPACE
            )

        self._nets = {
            "encoder_2d" : self._encoder_2d,
            "encoder_3d" : self._encoder_3d,
            "decoder_2d" : self._decoder_2d,
            "decoder_3d" : self._decoder_3d
        }

        if FLAGS.TRAINING:
            for net in self._nets: self._nets[net].train(True)

        self._criterion = torch.nn.MSELoss(reduction='mean')

        self._log_keys = ['loss_2D_AE', 'loss_3D_AE']
        # self._log_keys = ['loss_2D_AE', 'loss_3D_AE', 'latent-loss']

    def initialize(self, io_only=False):

        FLAGS.dump_config()


        self._initialize_io()



        if io_only:
            return

        self.init_network()


        for net in self._nets:
            n_trainable_parameters = 0
            for var in self._nets[net].parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            print(f"Total number of trainable parameters in {net}: {n_trainable_parameters}")

        self.init_optimizer()

        self.init_saver()

        self._global_step = 0


        state = self.restore_model()

        if state is not None:
            self.load_state(state)

        # If using half precision on the model, convert it now:
        if FLAGS.MODEL_HALF_PRECISION:
            self._net.half()

        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            for net in self._nets:
                self._nets[net].cuda()




    def init_optimizer(self):

        all_parameters = itertools.chain(
            self._nets['encoder_2d'].parameters(),
            self._nets['encoder_3d'].parameters(),
            self._nets['decoder_2d'].parameters(),
            self._nets['decoder_3d'].parameters(),
        )
        if FLAGS.OPTIMIZER == "adam":
            # Create an optimizer:
            if FLAGS.LEARNING_RATE <= 0:
                self._opt = torch.optim.Adam(all_parameters)
            else:
                self._opt = torch.optim.Adam(all_parameters, FLAGS.LEARNING_RATE)
        else:
            # Create an optimizer:
            if FLAGS.LEARNING_RATE <= 0:
                self._opt = torch.optim.SGD(all_parameters)
            else:
                self._opt = torch.optim.SGD(all_parameters, FLAGS.LEARNING_RATE)







    def init_saver(self):

        # This sets up the summary saver:
        self._saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY)

        if FLAGS.AUX_FILE is not None and FLAGS.TRAINING:
            self._aux_saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY + "/test/")
        elif FLAGS.AUX_FILE is not None and not FLAGS.TRAINING:
            self._aux_saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY + "/val/")
        else:
            self._aux_saver = None
        # This code is supposed to add the graph definition.
        # It doesn't currently work
        # temp_dims = list(dims['image'])
        # temp_dims[0] = 1
        # dummy_input = torch.randn(size=tuple(temp_dims), requires_grad=True)
        # self._saver.add_graph(self._net, (dummy_input,))

        # Here, either restore the weights of the network or initialize it:


    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        _, checkpoint_file_path = self.get_model_filepath()

        if not os.path.isfile(checkpoint_file_path):
            return None
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    print("Restoring weights from ", chkp_file)
                    break

        state = torch.load(chkp_file)
        return state

    def load_state(self, state):

        self._net.load_state_dict(state['state_dict'])
        self._opt.load_state_dict(state['optimizer'])
        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if FLAGS.COMPUTE_MODE == "GPU":
            for state in self._opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        return True

    def save_model(self):
        '''Save the model to file

        '''

        current_file_path, checkpoint_file_path = self.get_model_filepath()

        # save the model state into the file path:
        state_dict = {
            'global_step' : self._global_step,
            'state_dict'  : self._net.state_dict(),
            'optimizer'   : self._opt.state_dict(),
        }

        # Make sure the path actually exists:
        if not os.path.isdir(os.path.dirname(current_file_path)):
            os.makedirs(os.path.dirname(current_file_path))

        torch.save(state_dict, current_file_path)

        # Parse the checkpoint file to see what the last checkpoints were:

        # Keep only the last 5 checkpoints
        n_keep = 5


        past_checkpoint_files = {}
        try:
            with open(checkpoint_file_path, 'r') as _chkpt:
                for line in _chkpt.readlines():
                    line = line.rstrip('\n')
                    vals = line.split(":")
                    if vals[0] != 'latest':
                        past_checkpoint_files.update({int(vals[0]) : vals[1].replace(' ', '')})
        except:
            pass


        # Remove the oldest checkpoints while the number is greater than n_keep
        while len(past_checkpoint_files) >= n_keep:
            min_index = min(past_checkpoint_files.keys())
            file_to_remove = os.path.dirname(checkpoint_file_path) + "/" + past_checkpoint_files[min_index]
            os.remove(file_to_remove)
            past_checkpoint_files.pop(min_index)



        # Update the checkpoint file
        with open(checkpoint_file_path, 'w') as _chkpt:
            _chkpt.write('latest: {}\n'.format(os.path.basename(current_file_path)))
            _chkpt.write('{}: {}\n'.format(self._global_step, os.path.basename(current_file_path)))
            for key in past_checkpoint_files:
                _chkpt.write('{}: {}\n'.format(key, past_checkpoint_files[key]))


    def get_model_filepath(self):
        '''Helper function to build the filepath of a model for saving and restoring:


        '''

        # Find the base path of the log directory
        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path



    def _calculate_loss(self, minibatch_data, forward_results):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''

        # We calculate 3 losses:


        loss_2D_AE = self._criterion(input=forward_results['decoded_2d'], target=minibatch_data['data2d'])
        loss_3D_AE = self._criterion(input=forward_results['decoded_3d'], target=minibatch_data['data3d'])

        # latent_separation = forward_results['encoded_2d'] - forward_results['encoder_3d']

        # latent_loss = self._criterion(input=latent_separation,target = torch.zeros(latent_separation.shape))

        loss = {
            'loss_2D_AE'  : loss_2D_AE,
            'loss_3D_AE'  : loss_3D_AE,
            # 'latent_loss' : latent_loss,
        }

        loss['total_loss'] = loss_2D_AE + loss_3D_AE
        # loss['total_loss'] = loss_2D_AE + loss_3D_AE + latent_loss

        return loss


    def _compute_metrics(self, forward_results, loss):

        # Call all of the functions in the metrics dictionary:
        metrics = {}

        for key in loss:
            metrics[key] = loss[key] / FLAGS.LOSS_SCALE

        # metrics['loss']     = loss.data / FLAGS.LOSS_SCALE
        # accuracy = self._calculate_accuracy(logits, labels)
        # metrics.update(accuracy)


        return metrics

    def log(self, metrics, saver=''):


        if self._global_step % FLAGS.LOGGING_ITERATION == 0:

            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if self._log_keys != []:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self._log_keys])
            else:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])


            try:
                s += " ({:.2}s / {:.2} IOs / {:.2})".format(
                    (self._current_log_time - self._previous_log_time).total_seconds(),
                    metrics['io_fetch_time'],
                    metrics['step_time'])
            except:
                pass

            self._previous_log_time = self._current_log_time

            print("{} Step {} metrics: {}".format(saver, self._global_step, s))



    def summary(self, metrics,saver=""):

        if self._global_step % FLAGS.SUMMARY_ITERATION == 0:
            for metric in metrics:
                name = metric
                if saver == "test":
                    self._aux_saver.add_scalar(metric, metrics[metric], self._global_step)
                else:
                    self._saver.add_scalar(metric, metrics[metric], self._global_step)


            # try to get the learning rate
            self._saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            return


    def summary_images(self, forward_results, saver=""):

        return

        # if self._global_step % 1 * FLAGS.SUMMARY_ITERATION == 0:
        if self._global_step % 25 * FLAGS.SUMMARY_ITERATION == 0:

            # print(logits_image.shape)
            # print(labels_image.shape)
            logits_by_plane = torch.chunk(logits_image[0], chunks=FLAGS.NPLANES,dim=1)
            labels_by_plane = torch.chunk(labels_image[0], chunks=FLAGS.NPLANES,dim=0)
            # print(logits_by_plane[0].shape)
            # print(logits_by_plane[1].shape)
            # print(logits_by_plane[2].shape)


            for plane in range(FLAGS.NPLANES):
                val, prediction = torch.max(logits_by_plane[plane], dim=0)
                # This is a reshape and H/W swap:
                prediction = prediction.view(
                    [1, prediction.shape[-2], prediction.shape[-1]]
                    ).float()



                #TODO - need to address this function here!!!


                labels = labels_by_plane[plane].view(
                    [1, labels_by_plane[plane].shape[-2], labels_by_plane[plane].shape[-1]]
                    )
                # The images are in the format (Plane, W, H)
                # Need to transpose the last two dims in order to meet the (CHW) ordering
                # of tensorboardX


                # Values get mapped to gray scale, so put them in the range (0,1)
                labels[labels == 1] = 0.5
                labels[labels == 2] = 1.0

                prediction[prediction == 1] = 0.5
                prediction[prediction == 2] = 1.0


                if saver == "test":
                    self._aux_saver.add_image("prediction/plane_{}".format(plane),
                        prediction, self._global_step)
                    self._aux_saver.add_image("label/plane_{}".format(plane),
                        labels, self._global_step)

                else:
                    self._saver.add_image("prediction/plane_{}".format(plane),
                        prediction, self._global_step)
                    self._saver.add_image("label/plane_{}".format(plane),
                        labels, self._global_step)

        return

    def fetch_next_batch(self, mode='primary', metadata=False, force_pop=False):

        # if not FLAGS.SYNTHETIC:
            # metadata=True


        pop = True
        if self._iteration == 0 and not force_pop:
            pop = False

        minibatch_data = self._larcv_interface.fetch_minibatch_data(mode, pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])


        # We compute the weights upstream of converting to scn format:


        minibatch_data['data2d']  = data_transforms.larcvsparse_to_dense_2d(
            minibatch_data['data2d'], dense_shape=FLAGS.RAW_SHAPE_2D)
        minibatch_data['data3d']  = data_transforms.larcvsparse_to_dense_3d(
            minibatch_data['data3d'], dense_shape=FLAGS.RAW_SHAPE_3D)

        # Begin preloading the next data:
        self._larcv_interface.prepare_next(mode)

        return minibatch_data



    def increment_global_step(self):

        previous_epoch = int((self._global_step * FLAGS.MINIBATCH_SIZE) / self._epoch_size)
        self._global_step += 1
        current_epoch = int((self._global_step * FLAGS.MINIBATCH_SIZE) / self._epoch_size)

        self.on_step_end()

        if previous_epoch != current_epoch:
            self.on_epoch_end()

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass


    def to_torch(self, minibatch_data, device=None):

        # Convert the input data to torch tensors
        if FLAGS.COMPUTE_MODE == "GPU":
            if device is None:
                device = torch.device('cuda')
        else:
            if device is None:
                device = torch.device('cpu')

        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            else:
                # minibatch_data[key] = torch.tensor(minibatch_data[key], device=device)
                minibatch_data[key] = torch.tensor(minibatch_data[key], device=device)
                if FLAGS.INPUT_HALF_PRECISION:
                    minibatch_data[key] = minibatch_data[key].half()
        return minibatch_data

    def preprocess(self, minibatch_data):

        minibatch_data['data2d'] = torch.nn.functional.avg_pool2d(
            input       = minibatch_data['data2d'],
            kernel_size = [4,4],
            stride      = [4,4],
            padding     = [0,0]
        )

        minibatch_data['data3d'] = torch.nn.functional.avg_pool3d(
            input       = minibatch_data['data3d'],
            kernel_size = [4,4,4],
            stride      = [4,4,4],
            padding     = [0,0,0]
        )
        return minibatch_data


    def forward_pass(self, minibatch_data):

        minibatch_data = self.to_torch(minibatch_data)

        # This function performs downsamples as necessary:
        minibatch_data = self.preprocess(minibatch_data)

        forward_results = {}

        # We return several pieces from the forward pass:
        forward_results['encoded_2d'] = self._nets['encoder_2d'](minibatch_data['data2d'])

        forward_results['encoded_3d'] = self._nets['encoder_3d'](minibatch_data['data3d'])

        print("Encoded 2D shape: ", forward_results['encoded_2d'].shape)
        print("Encoded 3D shape: ", forward_results['encoded_3d'].shape)

        forward_results['decoded_2d'] = self._nets['decoder_2d'](forward_results['encoded_2d'])

        forward_results['decoded_3d'] = self._nets['decoder_3d'](forward_results['encoded_3d'])

        # We also compute an interesting piece, which is the decoded 3D image from the 2D space:
        forward_results['decoded_2d_to_3d'] = self._nets['decoder_3d'](forward_results['encoded_2d'])

        return forward_results

    def train_step(self):


        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.


        global_start_time = datetime.datetime.now()

        for net in self._nets:
            self._nets[net].train()

        # Reset the gradient values for this step:
        self._opt.zero_grad()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        minibatch_data = self.fetch_next_batch()
        io_end_time = datetime.datetime.now()

        forward_results = self.forward_pass(minibatch_data)


        verbose = False

        if verbose: print("Completed Forward pass")
        # Compute the loss based on the logits


        loss = self._calculate_loss(minibatch_data, forward_results)
        if verbose: print("Completed loss")

        # Compute the gradients for the network parameters:
        loss['total_loss'].backward()

        # If the loss is scaled, we have to un-scale after the backwards pass
        if FLAGS.LOSS_SCALE != 1.0:
            for param in self._net.parameters():
                param.grad /= FLAGS.LOSS_SCALE

        if verbose: print("Completed backward pass")


        # Compute any necessary metrics:
        metrics = self._compute_metrics(forward_results, loss)



        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = FLAGS.MINIBATCH_SIZE / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

        if verbose: print("Calculated metrics")



        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        self._opt.step()
        if verbose: print("Updated Weights")
        global_end_time = datetime.datetime.now()

        metrics['step_time'] = (global_end_time - step_start_time).total_seconds()


        self.log(metrics, saver="train")

        if verbose: print("Completed Log")

        self.summary(metrics, saver="train")
        self.summary_images(forward_results, saver="train")
        if verbose: print("Summarized")


        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Increment the global step value:
        self.increment_global_step()

        return

    def val_step(self):

        # First, validation only occurs on training:
        if not FLAGS.TRAINING: return

        # Second, validation can not occur without a validation dataloader.
        if FLAGS.AUX_FILE is None: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator
        if self._global_step != 0 and self._global_step % FLAGS.AUX_ITERATION == 0:

            self._net.eval()
            # Fetch the next batch of data with larcv
            # (Make sure to pull from the validation set)
            io_start_time = datetime.datetime.now()
            minibatch_data = self.fetch_next_batch('aux')
            io_end_time = datetime.datetime.now()

            forward_results = self.forward_pass(minibatch_data)

            # Compute the loss based on the logits
            loss = self._calculate_loss(minibatch_data, forward_results)


            # Compute any necessary metrics:
            metrics = self._compute_metrics(forward_results, loss)


            self.log(metrics, saver="test")
            self.summary(metrics, saver="test")
            self.summary_images(forward_results, saver="test")

            if not FLAGS.DISTRIBUTED:
                self._larcv_interface.next('aux')

            return



    def stop(self):
        # Mostly, this is just turning off the io:
        self._larcv_interface.stop()

    def checkpoint(self):

        if self._global_step % FLAGS.CHECKPOINT_ITERATION == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def ana_step(self):

        # First, validation only occurs on training:
        if FLAGS.TRAINING: return

        # perform a validation step

        # Set network to eval mode
        self._net.eval()
        # self._net.train()

        # Fetch the next batch of data with larcv
        minibatch_data = self.fetch_next_batch(metadata=True)

        # Convert the input data to torch tensors
        minibatch_data = self.to_torch(minibatch_data)


        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            forward_results = self.forward_pass(minibatch_data)



        # If there is an aux file, for ana that means an output file.
        # Call the larcv interface to write data:
        if FLAGS.AUX_FILE is not None:

            # To use the PyUtils class, we have to massage the data
            features = (logits.features).cpu()
            coords   = (logits.get_spatial_locations()).cpu()
            coords = coords[:,0:-1]
            # print("Features shape: ", features.shape)
            # print("Coords shape: ", coords.shape)

            # Compute the softmax:
            features = torch.nn.Softmax(dim=1)(features)
            val, prediction = torch.max(features, dim=-1)
            # print("Prediction shape: ", prediction.shape)

            # Assuming batch size of 1 here so we don't need to fiddle with the batch dimension.


            # We store the prediction for each plane, as well as it's 3 scores, seperately.
            # Each type, though (bkg/cosmic/neut) is rolled up into one producer

            list_of_dicts_by_label = {
                0 : [None] * FLAGS.NPLANES,
                1 : [None] * FLAGS.NPLANES,
                2 : [None] * FLAGS.NPLANES,
                'pred' : [None] * FLAGS.NPLANES,
            }

            for plane in range(FLAGS.NPLANES):
                locs = coords[:,0] == plane
                # print("Locs shape: ", locs.shape)
                this_coords = coords[locs]
                this_features = features[locs]

                # print("Sub coords shape: ", this_coords.shape)
                # print("Sub features shape: ", this_features.shape)

                # Ravel the cooridinates into flat indexes:
                indexes = self._y_spatial_size * this_coords[:,1] + this_coords[:,2]
                meta = [0, 0,
                        self._y_spatial_size, self._x_spatial_size,
                        self._y_spatial_size, self._x_spatial_size,
                        plane,
                    ]
                # print("Indexes shape: ", indexes.shape)

                for feature_type in [0,1,2]:
                    writeable_features = this_features[:, feature_type]
                    # print("Write features shape: ", writeable_features.shape)

                    list_of_dicts_by_label[feature_type][plane] = {
                        'value' : numpy.asarray(writeable_features).flatten(),
                        'index' : numpy.asarray(indexes.flatten()),
                        'meta'  : meta
                    }


                # Also do the prediction:
                this_prediction = prediction[locs]
                # print("Sub prediction shape: ", this_prediction.shape)
                list_of_dicts_by_label['pred'][plane] = {
                    'value' : numpy.asarray(this_prediction).flatten(),
                    'index' : numpy.asarray(indexes.flatten()),
                    'meta'  : meta
                }


            for l in [0,1,2]:
                self._larcv_interface.write_output(data=list_of_dicts_by_label[l],
                    datatype='sparse2d', producer='label_{}'.format(l),
                    entries=minibatch_data['entries'],
                    event_ids=minibatch_data['event_ids'])

            self._larcv_interface.write_output(data=list_of_dicts_by_label['pred'],
                datatype='sparse2d', producer='prediction'.format(l),
                entries=minibatch_data['entries'],
                event_ids=minibatch_data['event_ids'])

        # If the input data has labels available, compute the metrics:
        if 'label' in minibatch_data:
            # Compute the loss
            loss = self._calculate_loss(minibatch_data, forward_results)

            # Compute the metrics for this iteration:
            print("computing metrics for entry ", minibatch_data['entries'][0])
            metrics = self._compute_metrics(forward_results, loss)


            self.log(metrics, saver="ana")
            # self.summary(metrics, saver="test")
            # self.summary_images(forward_results, saver="ana")

        self._larcv_interface.next('aux')

        return

    def batch_process(self):

        # At the begining of batch process, figure out the epoch size:
        self._epoch_size = self._larcv_interface.size('primary')

        # This is the 'master' function, so it controls a lot


        # Run iterations
        for self._iteration in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break


            if FLAGS.TRAINING:
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step()


        if self._saver is not None:
            self._saver.close()
        if self._aux_saver is not None:
            self._aux_saver.close()
