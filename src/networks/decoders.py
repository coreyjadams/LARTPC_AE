
import torch
import torch.nn as nn

from . import utils


class decoder2D(torch.nn.Module):

    def __init__(self, *, n_planes, n_initial_filters, batch_norm, use_bias, residual, depth, blocks_per_layer):
        torch.nn.Module.__init__(self)

        self.n_planes = n_planes

        self.shaping_operation = None

        self.plane_decoders = []
        for p in range(n_planes):
            self.plane_decoders.append(
                plane_decoder2D(
                    n_initial_filters = n_initial_filters,
                    batch_norm        = batch_norm,
                    use_bias          = use_bias,
                    residual          = residual,
                    depth             = depth,
                    blocks_per_layer  = blocks_per_layer)
                )
            self.add_module(f'plane_encoder_{p}', self.plane_decoders[-1])


    def forward(self, x):
        # print("Decode 2D initial shape: ", x.shape)

        if self.shaping_operation is not None:
            x = self.shaping_operation(x)

        x = list(torch.chunk(x, self.n_planes, dim=3))

        x = [torch.squeeze(_x, dim=3) for _x in x ]

        for i in range(self.n_planes):
            x[i] = self.plane_decoders[i](x[i])


        # Recombine all images together, and remove the unneeded dimension:
        decoded_image = torch.stack(x, dim=1)
        decoded_image = torch.squeeze(decoded_image, dim=2)

        # print("Decode 2D final shape: ", decoded_image.shape)


        return decoded_image


class plane_decoder2D(torch.nn.Module):

    def __init__(self, *, n_initial_filters, batch_norm, use_bias, residual, depth, blocks_per_layer):
        torch.nn.Module.__init__(self)

        # At first glance, the decoders are resnets with configurable depth, run in reverse

        # self.initial_convolution = utils.Block2D(
        #     inplanes    = n_initial_filters,
        #     outplanes   = n_initial_filters,
        #     batch_norm  = batch_norm,
        #     use_bias    = use_bias)

        current_n_planes = n_initial_filters

        self.layers = []
        self.upsample_layers = []

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
            self.upsample_layers.append(
                utils.ConvolutionUpsample2D(
                    inplanes    = current_n_planes,
                    outplanes   = int(current_n_planes / 2),
                    batch_norm  = batch_norm,
                    use_bias    = use_bias,
                )
            )
            current_n_planes = int(current_n_planes / 2)
            self.add_module('downsample_{}'.format(i_layer), self.upsample_layers[-1])

        self.final_layer = utils.Convolution2D(
                inplanes    = current_n_planes,
                outplanes   = 1,
                batch_norm  = batch_norm,
                use_bias    = use_bias
        )

    def forward(self, x):


        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)

        x = self.final_layer(x)
        return x

class decoder3D(torch.nn.Module):

    def __init__(self, *, n_initial_filters, batch_norm, use_bias, residual, depth, blocks_per_layer):
        torch.nn.Module.__init__(self)

        # At first glance, the decoders are resnets with configurable depth, etc.

        # self.initial_convolution = utils.Block3D(
        #     inplanes    = n_initial_filters,
        #     outplanes   = n_initial_filters,
        #     batch_norm  = batch_norm,
        #     use_bias    = use_bias)

        current_n_planes = n_initial_filters

        self.layers = []
        self.upsample_layers = []

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
            self.upsample_layers.append(
                utils.ConvolutionUpsample3D(
                    inplanes    = current_n_planes,
                    outplanes   = int(current_n_planes/2),
                    batch_norm  = batch_norm,
                    use_bias    = use_bias,
                )
            )
            current_n_planes = int(current_n_planes / 2)
            self.add_module('downsample_{}'.format(i_layer), self.upsample_layers[-1])

        self.final_layer = utils.Convolution3D(
                inplanes    = current_n_planes,
                outplanes   = 1,
                batch_norm  = batch_norm,
                use_bias    = use_bias
        )


    def forward(self, x):

        # print("Decode 3D initial shape: ", x.shape)
        # return x

        # x = self.initial_convolution(inputs)

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)

        x = self.final_layer(x)


        # print("Decode 3D final shape: ", x.shape)


        return x
