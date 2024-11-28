import torch.nn as nn
import torch.nn.functional as F
import torch
from . import simple_transformer


class PositionEncoding2D(nn.Module):
    def __init__(self, max_size, dim):
        super(PositionEncoding2D, self).__init__()
        self.dim = dim
        self.max_size = max_size
        self.x_emb = nn.Embedding(max_size*2, dim)
        self.y_emb = nn.Embedding(max_size*2, dim)
        pass

    def forward(self, size):
        x = torch.arange(size, device=self.x_emb.weight.device)
        x = torch.tile(x, [size])
        x = x.reshape(-1)
        y = torch.arange(size, device=self.y_emb.weight.device)
        y = y.reshape(-1, 1)
        y = torch.tile(y, [1, size])
        y = y.reshape(-1)
        x_diff = torch.unsqueeze(x, 0) - torch.unsqueeze(x, 1)
        y_diff = torch.unsqueeze(y, 0) - torch.unsqueeze(y, 1)
        x_diff = torch.clamp(x_diff, -self.max_size, self.max_size)
        y_diff = torch.clamp(y_diff, -self.max_size, self.max_size)
        x_diff = x_diff + self.max_size
        y_diff = y_diff + self.max_size
        x_encoding = self.x_emb(x_diff)
        y_encoding = self.y_emb(y_diff)

        return x_encoding + y_encoding


class ImageSegmenter(nn.Module):
    def __init__(self, num_classes, image_size=1024, decoder_size=32, encoder_dim=256, decoder_dim=32):
        super(ImageSegmenter, self).__init__()
        self.image_size = image_size
        self.encoder_size = image_size // decoder_size
        self.decoder_size = decoder_size
        self.decoder_dim = decoder_dim
        self.embed_layer = nn.Conv2d(3, encoder_dim, 32, 32)
        encoder_num_heads = 8
        encoder_head_size = encoder_dim // encoder_num_heads
        decoder_num_heads = 4
        decoder_head_size = decoder_dim // decoder_num_heads
        self.encoder = simple_transformer.MultiLayerTransformer(
            4, encoder_dim, encoder_dim*4, encoder_head_size, encoder_num_heads, 0.1, 0.1, None, True)
        self.decoder = simple_transformer.MultiLayerTransformer(
            2, decoder_dim, decoder_dim*4, decoder_head_size, decoder_num_heads, 0.1, 0.1, None, True)
        self.encoder_position_encoding = PositionEncoding2D(self.encoder_size, encoder_head_size)
        self.decoder_position_encoding = PositionEncoding2D(self.decoder_size, decoder_head_size)
        self.decoder_embed_layer = nn.Linear(3, decoder_dim)
        self.pad0_position_embedding = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(1, 1, decoder_head_size)))
        self.pad1_position_embedding = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(1, 1, decoder_head_size)))
        self.classifier = nn.Linear(decoder_dim, num_classes)

        self.classifer_linear = nn.Linear(encoder_dim, decoder_size*decoder_size*num_classes)
        pass

    def forward_decoder_transformer(self, img, x, h, w):
        patches = F.unfold(img, self.decoder_size, stride=self.decoder_size)
        # patches: [B, 3*32*32, h*w]
        patches = torch.permute(patches, [0, 2, 1])
        # patches: [B, h*w, 3*32*32]
        patches = torch.reshape(patches, [patches.shape[0], -1, self.decoder_size*self.decoder_size, 3])
        # patches: [B, h*w, 32*32, 3]
        patches = self.decoder_embed_layer(patches)
        # patches: [B, h*w, 32*32, decoder_dim]
        num_enc_reshape = x.shape[-1] // self.decoder_dim
        x = torch.reshape(x, [x.shape[0], -1, num_enc_reshape, self.decoder_dim])
        # x: [B, h*w, num_enc_reshape, decoder_dim]
        pos_dec = self.decoder_position_encoding(self.decoder_size)
        # pos_dec: [decoder_size**2, decoder_size**2, decoder_dim]
        decoder_size2 = self.decoder_size * self.decoder_size

        pos_pad0 = torch.tile(self.pad0_position_embedding, [num_enc_reshape, num_enc_reshape, 1])
        pos_pad1 = torch.tile(self.pad1_position_embedding, [decoder_size2, num_enc_reshape, 1])
        
        # pad decoder position encoding
        pos_pad = torch.cat([pos_pad0, pos_pad1], axis=0)
        # pos_pad: [num_enc_reshape+decoder_size2, num_enc_reshape, decoder_dim]

        pos_dec = torch.cat([pos_pad1.transpose(0, 1), pos_dec], axis=0)
        # pos_dec: [num_enc_reshape+decoder_size2, decoder_size2, decoder_dim]

        pos_dec = torch.cat([pos_pad, pos_dec], axis=1)
        # pos_dec: [num_enc_reshape+decoder_size2, num_enc_reshape+decoder_size2, decoder_dim]

        x = torch.cat([x, patches], axis=2)
        # x: [B, h*w, num_enc_reshape+decoder_size2, decoder_dim]

        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        # x: [B*h*w, num_enc_reshape+decoder_size2, decoder_dim]

        x = self.decoder(x, structure_matrix=pos_dec)
        # x: [B*h*w, num_enc_reshape+decoder_size2, decoder_dim]

        x = x[:, num_enc_reshape:, :]
        # x: [B*h*w, decoder_size**2, decoder_dim]
        x = x.reshape(-1, h, w, self.decoder_size, self.decoder_size, self.decoder_dim)
        # x: [B, h, w, decoder_size, decoder_size, decoder_dim]
        x = torch.permute(x, [0, 5, 3, 4, 1, 2])
        # x: [B, decoder_dim, decoder_size, decoder_size, h, w]
        x = torch.reshape(x, [x.shape[0], x.shape[1]*decoder_size2, -1])
        # x: [B, decoder_dim*decoder_size2, h*w]
        x = F.fold(x, [h*self.decoder_size, w*self.decoder_size], self.decoder_size, stride=self.decoder_size)
        # x: [B, decoder_dim, H, W]

        x = self.classifier(x)

        return x
    
    def forward_decoder_linear(self, img, x, h, w):
        x = self.classifer_linear(x)
        # x: [B, h*w, decoder_size2*num_classes]
        x = x.permute([0, 2, 1]).contiguous()
        # x: [B, decoder_size2*num_classes, h*w]

        x = F.fold(x, [h*self.decoder_size, w*self.decoder_size], self.decoder_size, stride=self.decoder_size)
        # x: [B, num_classes, H, W]

        return x
        pass

    def forward(self, img):
        # img: [B, C, H, W]
        x = self.embed_layer(img)
        # x: [B, 256, H/32, W/32]

        x = torch.permute(x, [0, 2, 3, 1])
        # x: [B, h, w, c]

        h = x.shape[1]
        w = x.shape[2]

        x = torch.reshape(x, [x.shape[0], -1, x.shape[-1]])
        # x: [B, h*w, c]

        pos_enc = self.encoder_position_encoding(self.encoder_size)
        # pos_enc: [h*w, h*w, c]
        x = self.encoder(x, structure_matrix=pos_enc)
        # x: [B, h*w, 256]

        # return self.forward_decoder_transformer(img, x, h, w)
        return self.forward_decoder_linear(img, x, h, w)
    
    def compute_loss(self, pixel_scores, mask):
        # pixel_scores: [B, num_classes, H, W]
        # mask: [B, H, W]
        loss = F.cross_entropy(pixel_scores, mask)
        return loss
    pass