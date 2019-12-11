
import torch
import torch.nn as nn

from . import utils


class encoder2D(torch.nn.Module):

    def __init__(self, *, n_planes, n_initial_filters, batch_norm, use_bias, residual, depth, blocks_per_layer):
        torch.nn.Module.__init__(self)

        self.n_planes = n_planes

        self.plane_encoders = []
        for p in range(n_planes):
            self.plane_encoders.append(
                plane_encoder2D(
                    n_initial_filters = n_initial_filters,
                    batch_norm        = batch_norm,
                    use_bias          = use_bias,
                    residual          = residual,
                    depth             = depth,
                    blocks_per_layer  = blocks_per_layer)
                )
            self.add_module(f'plane_encoder_{p}', self.plane_encoders[-1])

        self.shaping_operation = None

    def forward(self, x):
        # print("Initial input shape: ", x.shape)

        x = list(torch.chunk(x, self.n_planes, dim=1))
        for i in range(self.n_planes):
            x[i] = self.plane_encoders[i](x[i])


        encoded_image = torch.stack(x, dim=3)

        if self.shaping_operation is not None:
            encoded_image = self.shaping_operation(encoded_image)

        return encoded_image

class plane_encoder2D(torch.nn.Module):

    def __init__(self, *, n_initial_filters, batch_norm, use_bias, residual, depth, blocks_per_layer):
        torch.nn.Module.__init__(self)


        # At first glance, the encoders are resnets with configurable depth, etc.

        self.initial_convolution = utils.Convolution2D(
            inplanes    = 1,
            outplanes   = n_initial_filters,
            batch_norm  = batch_norm,
            use_bias    = use_bias)

        current_n_planes = n_initial_filters

        self.layers = []
        self.downsample_layers = []

        for i_layer in range(depth):
            self.layers.append(
                utils.BlockSeries2D(
                    inplanes    = current_n_planes,
                    n_blocks    = blocks_per_layer,
                    batch_norm  = batch_norm,
                    use_bias    = use_bias,
                    residual    = residual
                )
            )

            self.add_module('blockseries_{}'.format(i_layer), self.layers[-1])
            # Convolutional downsample:
            self.downsample_layers.append(
                utils.ConvolutionDownsample2D(
                    inplanes    = current_n_planes,
                    outplanes   = current_n_planes*2,
                    batch_norm  = batch_norm,
                    use_bias    = use_bias,
                )
            )
            current_n_planes *= 2
            self.add_module('downsample_{}'.format(i_layer), self.downsample_layers[-1])

        # self.final_layer = torch.nn.Linear(in_features=12*12, out_features=FLAGS.LATENT_SIZE, bias=use_bias)

    def forward(self, inputs):

        x = self.initial_convolution(inputs)

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.downsample_layers[i](x)


        return x

        return self.final_layer(x)


class encoder3D(torch.nn.Module):

    def __init__(self, *, n_initial_filters, batch_norm, use_bias, residual, depth, blocks_per_layer):
        torch.nn.Module.__init__(self)


        # At first glance, the encoders are resnets with configurable depth, etc.

        self.initial_convolution = utils.Convolution3D(
            inplanes    = 1,
            outplanes   = n_initial_filters,
            batch_norm  = batch_norm,
            use_bias    = use_bias)

        current_n_planes = n_initial_filters

        self.layers = []
        self.downsample_layers = []

        for i_layer in range(depth):
            self.layers.append(
                utils.BlockSeries3D(
                    inplanes    = current_n_planes,
                    n_blocks    = blocks_per_layer,
                    batch_norm  = batch_norm,
                    use_bias    = use_bias,
                    residual    = residual
                )
            )

            self.add_module('blockseries_{}'.format(i_layer), self.layers[-1])
            # Convolutional downsample:
            self.downsample_layers.append(
                utils.ConvolutionDownsample3D(
                    inplanes    = current_n_planes,
                    outplanes   = current_n_planes*2,
                    batch_norm  = batch_norm,
                    use_bias    = use_bias,
                )
            )
            current_n_planes *= 2
            self.add_module('downsample_{}'.format(i_layer), self.downsample_layers[-1])

        # self.final_layer = torch.nn.Linear(in_features=12*12,, bias=use_bias)

    def forward(self, inputs):

        # print("Initial input shape: ", inputs.shape)

        x = self.initial_convolution(inputs)

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.downsample_layers[i](x)

        return x

        # return self.final_layer(x)
