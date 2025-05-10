import torch
import torch.nn as nn
from unet import UNET_OutputLayer, TimeEmbedding
from transformers import CLIPTokenizer
from scheduler import get_time_embedding
import model_converter2
from unet import UNET
import model_converter
from diffusion import Diffusion
from torch.nn import functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class ControlNet(nn.Module):
    def __init__(self, im_channels,
                 model_config,
                 model_locked=True,
                 model_ckpt=None,
                 device=None,
                 down_sample_factor=32):
        super(ControlNet, self).__init__()

        self.down_channels = model_config['down_channels']
        self.model_locked = model_locked
        self.emb_time_dim = model_config['time_emb_dim']
        self.latent = model_config['latent_dim']
        self.mid_channels = model_config['mid_channels']

        # self.time_emb_layer= TimeEmbedding(self.emb_time_dim)
        self.output_layer = UNET_OutputLayer(self.emb_time_dim, self.latent)


        # tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
        # self.model_locked = model_locked
        # self.trained_unet = UNET()
        #
        #
        # # Load weights for the trained model
        # if model_ckpt is not None and device is not None:
        #     print('Loading Trained Diffusion Model')
        #
        #     self.trained_unet.load_state_dict(model_ckpt['diffusion'], strict=True)

        print("Loading Trained Diffusion Model")

        # state_dict = model_converter2.load_from_standard_weights(model_ckpt, device)
        # self.trained_unet = UNET().to(device)
        # self.trained_unet.load_state_dict(state_dict['diffusion'], strict=True)

        state_dict = model_converter.load_from_standard_weights(model_ckpt, device)
        self.trained_unet = Diffusion().to(device)
        self.trained_unet.load_state_dict(state_dict['diffusion'], strict=True)
        # total_params = sum(p.numel() for p in self.trained_unet.parameters())
        # print(f"Params for Trained Unet: {total_params:,}")  # Should match known values
        # print(list(self.trained_unet.state_dict().keys())[:5])

        # self.trained_unet = model_ckpt['diffusion']
        # self.contorl_unet = UNET()
        # if model_ckpt is not None and device is not None:
        #     print('Loading Trained Diffusion Model')
        #
        #
        #     self.contorl_unet.load_state_dict(model_ckpt['diffusion'], strict=True)

        print("Loading Trained Diffusion model for ControlNet...")

        # self.control_unet = UNET().to(device)
        # self.control_unet.load_state_dict(state_dict['diffusion'], strict=True)

        self.control_unet = Diffusion().to(device)
        self.control_unet.load_state_dict(state_dict['diffusion'], strict=True)
        # total_params = sum(p.numel() for p in self.control_unet.parameters())
        # print(f"Params Controlnet Unet: {total_params:,}")  # Should match known values
        # print(list(self.control_unet.state_dict().keys())[:5])

        hint_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1 ),
            nn.SiLU(),
            nn.Conv2d( 16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d( 32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d( 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d( 96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
        )
        hint_block.append(nn.Sequential(
            nn.Conv2d(256, 320, kernel_size=3, padding=(1, 1)),
            nn.SiLU(),
            make_zero_module(nn.Conv2d(320,
                                       320, kernel_size=1, padding=0))
        ))

        self.control_unet_hint_block = nn.Sequential(*hint_block)
        down_zero_conv = nn.ModuleList([
            nn.Sequential(
                make_zero_module(nn.Conv2d(self.down_channels[0], self.down_channels[0], kernel_size=1, padding=0))
            )
                for _ in range(4)
        ])

        down_zero_conv.extend([
            nn.Sequential(
                make_zero_module(nn.Conv2d(self.down_channels[1], self.down_channels[1], kernel_size=1, padding=0))
            )
                for _ in range(3)
        ])

        down_zero_conv.extend([
            nn.Sequential(
                make_zero_module(nn.Conv2d(self.down_channels[2], self.down_channels[2], kernel_size=1, padding=0))
            )
                for _ in range(5)
        ])

        # Assign to self.unet_control_down_zero_conv as a ModuleList
        self.unet_control_down_zero_conv = down_zero_conv

        self.unet_control_bottneck_zero_conv = nn.ModuleList([
                make_zero_module(
                nn.Conv2d(self.mid_channels[0],
                          self.mid_channels[1],
                          kernel_size=1, padding=0)
                )
            for _ in range(3)
            ])

    def get_params(self):
        params = list(self.control_unet.parameters())
        params += list(self.control_unet_hint_block.parameters())
        params += list(self.unet_control_down_zero_conv.parameters())
        params += list(self.unet_control_bottneck_zero_conv.parameters())

        if not self.model_locked:
            params += list(self.trained_unet.unet.decoders.parameters())

        return params

    def forward(self, x, text, hints, t):
        # Check input data
        # print(x.shape)
        # print(x)
        # print(hints)
        # print(hints.shape)
        # print(text.shape)
        # print(text)
        assert not torch.isnan(x).any(), "Input x contains NaN values"
        assert not torch.isinf(x).any(), "Input x contains Inf values"
        assert not torch.isnan(hints).any(), "Input hints contains NaN values"
        assert not torch.isinf(hints).any(), "Input hints contains Inf values"

        get_time_emb_trained = get_time_embedding(t, self.emb_time_dim)
        trained_unet_t_emb = self.trained_unet.time_embedding(get_time_emb_trained)

        # print("trained_unet_t_emb shape:", trained_unet_t_emb.shape, trained_unet_t_emb)

        trained_unet_down_outs = []
        trained_unet_out = x

        with torch.no_grad():
            for layers in self.trained_unet.unet.encoders:
                trained_unet_out = layers(trained_unet_out, text, trained_unet_t_emb)
                trained_unet_down_outs.append(trained_unet_out)
                # assert not torch.isnan(trained_unet_out).any(), "NaN detected in encoder"

        control_get_time_emb = get_time_embedding(t, self.emb_time_dim)
        control_unet_t_emb = self.control_unet.time_embedding(control_get_time_emb)
        # print("control_unet_t_emb shape:", control_unet_t_emb.shape)

        control_unet_down_outs = []
        control_unet_out_hint = self.control_unet_hint_block(hints)
        # print("Control unet hint shape:", control_unet_out_hint.shape, control_unet_out_hint)

        control_unet_out = x
        # print("before Layers shape:", control_unet_out.shape)

        # Process the first SwitchSequential layer
        control_unet_out = self.control_unet.unet.encoders[0](control_unet_out, text, control_unet_t_emb)

        # print("after the first layer shape:", control_unet_out.shape)
        control_unet_out += control_unet_out_hint
        # print("control_unet_out += control_unet_out_hint shape:", control_unet_out.shape)

        # Iterate over the remaining encoder layers
        for i, layers in enumerate(self.control_unet.unet.encoders[1:]):
            zero_conv_out = self.unet_control_down_zero_conv[i](control_unet_out)
            control_unet_down_outs.append(zero_conv_out)
            control_unet_out = layers(control_unet_out, text, control_unet_t_emb)
            if i ==10:
                zero_conv_out = self.unet_control_down_zero_conv[i+1](control_unet_out)
                control_unet_down_outs.append(zero_conv_out)

            # assert not torch.isnan(control_unet_out).any(), "NaN detected in control encoder"

        # Bottleneck
        assert len(self.trained_unet.unet.bottleneck) == len(self.control_unet.unet.bottleneck) == len(
            self.unet_control_bottneck_zero_conv), \
            "Mismatch in the number of bottleneck layers"

        # Bottleneck processing in forward
        for index, layers in enumerate(self.trained_unet.unet.bottleneck):
            # Debugging: Print layer details
            # print(f"Bottleneck layer {index}: {layers}")
            # if hasattr(layers, 'forward'):
            #     print(f"Forward method arguments: {layers.forward.__code__.co_varnames}")
            #     Forwardmethodarguments: ('self', 'x', 'context', 'residue_long', 'n', 'c', 'h', 'w', 'residue_short', 'gate')

            # Process trained UNet bottleneck layer
            if 'context' in layers.forward.__code__.co_varnames:
                trained_unet_out = layers(trained_unet_out, context=text)
            else:
                trained_unet_out = layers(trained_unet_out, time=trained_unet_t_emb)

            # Process control UNet bottleneck layer
            control_layers = self.control_unet.unet.bottleneck[index]
            if 'context' in control_layers.forward.__code__.co_varnames:
                control_unet_out = control_layers(control_unet_out, context=text)
            else:
                control_unet_out = control_layers(control_unet_out, time=control_unet_t_emb)

            # Add control UNet output to trained UNet output (after zero convolution)
            trained_unet_out += self.unet_control_bottneck_zero_conv[index](control_unet_out)


        # trained_unet_out = self.trained_unet.unet.bottleneck(trained_unet_out, text, trained_unet_t_emb)
        # control_unet_out = self.control_unet.unet.bottleneck(control_unet_out, text, control_unet_t_emb)
        # trained_unet_out += control_unet_out
        # assert not torch.isnan(trained_unet_out).any(), "NaN detected in bottleneck"

        # Decoder
        assert len(trained_unet_down_outs) == len(control_unet_down_outs) == len(self.trained_unet.unet.decoders), \
            "Mismatch in the number of skip connections"
        for layers in self.trained_unet.unet.decoders:
            trained_unet_down_out = trained_unet_down_outs.pop()
            control_unet_down_out = control_unet_down_outs.pop()

            # Resize if necessary
            if control_unet_down_out.shape[2:] != trained_unet_down_out.shape[2:]:
                control_unet_down_out = F.interpolate(control_unet_down_out, size=trained_unet_down_out.shape[2:],
                                                      mode='bilinear', align_corners=False)

            trained_unet_out = torch.cat((trained_unet_out, control_unet_down_out + trained_unet_down_out), dim=1)
            trained_unet_out = layers(trained_unet_out, text, trained_unet_t_emb)
            # assert not torch.isnan(trained_unet_out).any(), "NaN detected in decoder"

        # Final layer
        trained_unet_out = self.trained_unet.final(trained_unet_out)
        # # assert not torch.isnan(trained_unet_out).any(), "NaN detected in final layer"
        # total_params = sum(p.numel() for p in self.trained_unet.parameters())
        # print(f"Params for Trained Unet: {total_params:,}")  # Should match known values
        # print(list(self.trained_unet.state_dict().keys())[:50])
        #
        # total_params = sum(p.numel() for p in self.control_unet.parameters())
        # print(f"Params Controlnet Unet: {total_params:,}")  # Should match known values
        # print(list(self.control_unet.state_dict().keys())[:50])

        return trained_unet_out
