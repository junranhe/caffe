import os
import sys

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2


def FireBlock(net, from_layer, block_name, out_sq, out_ex1, out_ex3):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    conv_name = "{}/squeeze1x1".format(block_name)
    net[conv_name] = L.Convolution(net[from_layer], num_output=out_sq, kernel_size=1, stride=1, **kwargs)
    sq_relu_name = "{}/relu_squeeze1x1".format(block_name)
    net[sq_relu_name] = L.ReLU(net[conv_name], in_place=True)

    expand_layers = []

    conv_name="{}/expand1x1".format(block_name)
    net[conv_name] = L.Convolution(net[sq_relu_name], num_output=out_ex1, kernel_size=1, stride=1, **kwargs)
    relu_name = "{}/relu_expand1x1".format(block_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)
    expand_layers.append(net[relu_name])

    conv_name="{}/expand3x3".format(block_name)
    net[conv_name] = L.Convolution(net[sq_relu_name], num_output=out_ex3, kernel_size=3, pad=1, stride=1, **kwargs)
    relu_name = "{}/relu_expand3x3".format(block_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)
    expand_layers.append(net[relu_name])

    concat_name = "{}/concat".format(block_name)
    net[concat_name] = L.Concat(*expand_layers, axis=1)

def SqueezeNetBody(net, from_layer, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    # conv1
    net['conv1'] = L.Convolution(net[from_layer], num_output=64, kernel_size=3, pad=1, stride=2, **kwargs)
    net['relu_conv1'] = L.ReLU(net['conv1'], in_place=True)
    # pool1
    net['pool1'] = L.Pooling(net['relu_conv1'], pool=P.Pooling.MAX, kernel_size=3, stride=2)
    # fire2
    FireBlock(net, 'pool1', 'fire2', 16, 64, 64)
    # fire3
    FireBlock(net, 'fire2/concat', "fire3", 16, 64, 64)
    # pool3, to make the outpue shape as 38x38, we set pad=1
    net['pool3'] = L.Pooling(net['fire3/concat'], pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=2)
    # fire4
    FireBlock(net, 'pool3', 'fire4', 32, 128, 128)
    # fire5
    FireBlock(net, 'fire4/concat', 'fire5', 32, 128, 128)
    # pool5
    net['pool5'] = L.Pooling(net['fire5/concat'], pool=P.Pooling.MAX, kernel_size=3, stride=2)
    # fire6
    FireBlock(net, 'pool5', 'fire6', 48, 192, 192)
    # fire7
    FireBlock(net, 'fire6/concat', 'fire7', 48, 192, 192)
    # fire8
    FireBlock(net, 'fire7/concat', 'fire8', 64, 256, 256)
    # fire9
    FireBlock(net, 'fire8/concat', 'fire9', 64, 256, 256)

    # Update freeze layers, set learning rate to 0
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net

pynetbuilder_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../pynetbuilder'
'''
    Class to stitch together a residual network
    Borrowed from pynetbuilder(https://github.com/jay-mahadeokar/pynetbuilder)
'''
class ResNet():

    def stitch(self, netspec, from_layer, is_train, source, main_branch, num_output_stage1, fc_layers, blocks):
        cwd = os.getcwd()
        os.chdir(pynetbuilder_dir)
        sys.path.append('netbuilder')
        from lego.hybrid import ConvBNReLULego, EltwiseReLULego, ShortcutLego, ConvBNLego
        from lego.data import ImageDataLego
        from lego.base import BaseLegoFunction, Config

        #netspec = caffe.NetSpec()

        if is_train:
            include = 'train'
            use_global_stats = False
            batch_size = 256
        else:
            include = 'test'
            use_global_stats = True
            batch_size = 1

        # Freeze 1st 2 stages and dont update batch norm stats
        Config.set_default_params('Convolution', 'param', [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        if is_train:
            use_global_stats = True


        """
        # Data layer, its dummy, you need to replace this with Annotated data layer
        params = dict(name='data', batch_size=1, ntop=2,
                      memory_data_param=dict(batch_size=1, channels=3, height=300, width=300)
                      )
        netspec.data, netspec.label = BaseLegoFunction('MemoryData', params).attach(netspec, [])
        """

        # Stage 1
        params = dict(name='1', num_output=64, kernel_size=7,
                      use_global_stats=use_global_stats, pad=3, stride=2)
        stage1 = ConvBNReLULego(params).attach(netspec, [netspec[from_layer]])
        params = dict(kernel_size=3, stride=2, pool=P.Pooling.MAX, name='pool1')
        pool1 = BaseLegoFunction('Pooling', params).attach(netspec, [stage1])

        num_output = num_output_stage1

        # Stages 2 - 5
        last = pool1
        for stage in range(4):
            name = str(stage + 1)

            for block in range(blocks[stage]):
                if block == 0:
                    shortcut = 'projection'
                    if stage > 0:
                        stride = 2
                    else:
                        stride = 1
                else:
                    shortcut = 'identity'
                    stride = 1

                # this is for resnet 18 / 34, where the first block of stage
                # 0 does not need projection shortcut
                if block == 0 and stage == 0 and main_branch == 'normal':
                    shortcut = 'identity'

                # This is for not downsampling while creating detection
                # network
                # if block == 0 and stage == 1:
                #    stride = 1

                name = 'stage' + str(stage) + '_block' + str(block)
                curr_num_output = num_output * (2 ** (stage))

                params = dict(name=name, num_output=curr_num_output,
                              shortcut=shortcut, main_branch=main_branch,
                              stride=stride, use_global_stats=use_global_stats,)
                last = ShortcutLego(params).attach(netspec, [last])

            # TODO: Note this should be configurable
            if stage == 0:
               Config.set_default_params('Convolution', 'param', [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
               if is_train:
                   use_global_stats = False


        '''
                You should modify these layers in order to experiment with different
                architectures specific for detection
        '''
        if not fc_layers:
            # Last stage
            pool_params = dict(kernel_size=7, stride=1, pool=P.Pooling.AVE, name='pool', pad=3)
            pool = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])
        else:

            '''pool_params = dict(name='pool_before1024', kernel_size=3, stride=1, pool=P.Pooling.MAX, pad=1)
            pool = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])
            conv_last1_params = dict(name='3by3_1024', num_output=1024, use_global_stats=use_global_stats,
                     # pad=0, kernel_size=1)
                     pad=1, kernel_size=3, dilation=3)
            conv_last1 = ConvBNReLULego(conv_last1_params).attach(netspec, [pool])
            '''

            conv_last1_params = dict(name='1by1_2048', num_output=2048, use_global_stats=use_global_stats,
                                 pad=1, kernel_size=1, dilation=1)
                                 # pad=3, kernel_size=7)
            conv_last1 = ConvBNReLULego(conv_last1_params).attach(netspec, [last])

            pool_last_params = dict(kernel_size=7, stride=1, pool=P.Pooling.AVE, name='pool', pad=3)
            pool_last = BaseLegoFunction('Pooling', pool_last_params).attach(netspec, [conv_last1])

            conv_last2_params = dict(name='1by1_4096', kernel_size=1, num_output=4096, use_global_stats=use_global_stats,
                                     stride=1, pad=0)
            conv_last2 = ConvBNReLULego(conv_last2_params).attach(netspec, [pool_last])

        os.chdir(cwd)  # change to previous working dir
        return netspec

def ResNet50Body(net, from_layer, params):
    """
    get resnet with pynetbuilder
    """
    is_train = params['is_train']
    main_branch = params['main_branch']
    num_output_stage1 = params['num_output_stage1']
    fc_layers = params['fc_layers']
    blocks = params['blocks']

    net = ResNet().stitch(net, from_layer, is_train=is_train, source='tt', main_branch=main_branch,
                              num_output_stage1=num_output_stage1, fc_layers=fc_layers, blocks=blocks)
    return net

def add_extra_layers_resnet(netspec, last, params):
    from lego.hybrid import ShortcutLego, ConvBNReLULego

    from lego.base import BaseLegoFunction

    blocks = params['extra_blocks']
    num_outputs = params['extra_num_outputs']
    is_train = params['is_train']
    main_branch = params['main_branch']

    use_global_stats = False if is_train else True


    for stage in range(len(blocks)):
        for block in range(blocks[stage]):
            if block == 0:
                shortcut = 'projection'
                stride = 2
            else:
                shortcut = 'identity'
                stride = 1

            name = 'stage' + str(stage + 4) + '_block' + str(block)
            curr_num_output = num_outputs[stage]

            params = dict(name=name, num_output=curr_num_output,
                          shortcut=shortcut, main_branch=main_branch,
                          stride=stride, use_global_stats=use_global_stats,
                          filter_mult=None)
            last = ShortcutLego(params).attach(netspec, [last])

    # Add global pooling layer.
    pool_param = dict(name='pool_last', pool=P.Pooling.AVE, global_pooling=True)
    pool = BaseLegoFunction('Pooling', pool_param).attach(netspec, [last])
    return netspec
