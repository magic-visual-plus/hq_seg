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


class PositionEncoding1D(nn.Module):
    def __init__(self, max_size, dim):
        super(PositionEncoding1D, self).__init__()
        self.dim = dim
        self.max_size = max_size
        self.emb = nn.Embedding(max_size*max_size, dim)
        pass

    def forward(self, size):
        x = torch.arange(size*size, device=self.emb.weight.device)
        return self.emb(x)
        pass

class MultiConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(MultiConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        pass

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        x1 = x + x1
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.activation(x2)
        return x2 + x1

class ImageSegmenterDecoderTransformer(nn.Module):
    def __init__(self, num_classes, decoder_dim, decoder_size):
        super(ImageSegmenterDecoderTransformer, self).__init__()
        self.decoder_dim = decoder_dim
        self.decoder_size = decoder_size
        self.decoder_embed_layer = nn.Linear(3, decoder_dim)
        decoder_num_heads = 8
        decoder_head_size = decoder_dim // decoder_num_heads

        self.decoder_position_encoding = PositionEncoding2D(self.decoder_size, decoder_head_size)

        self.pad0_position_embedding = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(1, 1, decoder_head_size)))
        self.pad1_position_embedding = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(1, 1, decoder_head_size)))

        self.transformer = simple_transformer.MultiLayerTransformer(
            2, decoder_dim, decoder_dim*4, decoder_head_size, decoder_num_heads, 0.1, 0.1, None, True)
        
        self.classifier = nn.Linear(decoder_dim, num_classes)
        pass

    def forward(self, x, patches):
        # x: [B, encoder_dim]
        # patches: [B, 3, decoder_size, decoder_size]

        patches = torch.permute(patches, [0, 2, 3, 1])
        # patches: [B, decoder_size, decoder_size, 3]
        patches_x = self.decoder_embed_layer(patches)
        # patches_x: [B, decoder_size, decoder_size, decoder_dim]
        patches_x = torch.reshape(patches_x, [patches_x.shape[0], -1, patches_x.shape[-1]])
        # patches_x: [B, decoder_size2, decoder_dim]
        num_enc_reshape = x.shape[-1] // self.decoder_dim
        x = torch.reshape(x, [x.shape[0], num_enc_reshape, self.decoder_dim])
        # x: [B, num_enc_reshape, decoder_dim]
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

        x = torch.cat([x, patches_x], axis=1)
        # x: [B, num_enc_reshape+decoder_size2, decoder_dim]

        x = self.transformer(x, structure_matrix=pos_dec)
        # x: [B*h*w, num_enc_reshape+decoder_size2, decoder_dim]

        x = x[:, num_enc_reshape:, :]
        # x: [B, decoder_size2, decoder_dim]

        x = self.classifier(x)
        # x: [B, decoder_size2, num_classes]

        x = torch.reshape(x, [x.shape[0], self.decoder_size, self.decoder_size, x.shape[-1]])
        # x: [B, decoder_size, decoder_size, num_classes]

        x = torch.permute(x, [0, 3, 1, 2])
        return x
        pass


class ImageSegmenter(nn.Module):
    @classmethod
    def load_from_file(cls, filename):
        data = torch.load(filename)
        model = cls(data['num_classes'])
        model.load_state_dict(data['state_dict'])
        return model

    def __init__(self, num_classes, image_size=1024, decoder_size=32, encoder_dim=512, decoder_dim=32):
        super(ImageSegmenter, self).__init__()
        self.image_size = image_size
        self.encoder_size = image_size // decoder_size
        self.decoder_size = decoder_size
        self.decoder_dim = decoder_dim
        # self.embed_layer = nn.Conv2d(3, encoder_dim, 32, 32)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=4, padding=0)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.multi_conv1 = MultiConv2d(128, 128, 3, 1, 1)

        encoder_num_heads = 16
        encoder_head_size = encoder_dim // encoder_num_heads
        self.encoder = simple_transformer.MultiLayerTransformer(
            4, encoder_dim, encoder_dim*4, encoder_head_size, encoder_num_heads, 0.1, 0.1, None, True)
        # self.decoder = ImageSegmenterDecoderTransformer(num_classes, decoder_dim, decoder_size)
        self.encoder_position_encoding = PositionEncoding2D(self.encoder_size, encoder_head_size)

        self.classifer_linear = nn.Linear(encoder_dim, decoder_size*decoder_size*num_classes)
        self.inv_conv1 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2, padding=0)
        self.inv_conv2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=4, padding=0)
        self.inv_conv3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4, padding=0)
        self.inv_bn1 = nn.BatchNorm2d(256)
        self.inv_bn2 = nn.BatchNorm2d(128)
        self.inv_multi_conv3 = MultiConv2d(64, 64, 3, 1, 1)
        self.inv_multi_conv1 = MultiConv2d(512, 512, 3, 1, 1)
        self.inv_multi_conv2 = MultiConv2d(256, 256, 3, 1, 1)
        self.classifier = nn.Conv2d(67, num_classes, kernel_size=3, stride=1, padding=1)
        pass

    def forward_decoder_transformer(self, img, x):
        # x: [B, h, w, encoder_dim]
        h = x.shape[1]
        w = x.shape[2]
        batch_size = x.shape[0]
        x = torch.reshape(x, [batch_size, -1, x.shape[-1]])
        patches = F.unfold(img, self.decoder_size, stride=self.decoder_size)
        # patches: [B, 3*decoder_size2, h*w]
        patches = torch.permute(patches, [0, 2, 1])
        # patches: [B, h*w, 3*32*32]
        patches = torch.reshape(patches, [-1, self.decoder_size*self.decoder_size, 3])
        # patches: [B*h*w, 32*32, 3]
        
        x = torch.reshape(x, [-1, x.shape[-1]])
        # x: [B*h*w, encoder_dim]

        scores = self.decoder(x, patches)
        # scores: [B*h*w, decoder_size2, num_classes]
        scores = torch.reshape(scores, [batch_size, h, w, scores.shape[-2], scores.shape[-1]])
        # scores: [B, h, w, decoder_size2, num_classes]
        scores = torch.permute(scores, [0, 4, 3, 1, 2])
        # scores: [B, num_classes, decoder_size2, h, w]
        scores = torch.reshape(scores, [batch_size, -1, h*w])
        # scores: [B, num_classes*decoder_size2, h*w]
        scores = F.fold(
            scores, [h*self.decoder_size, w*self.decoder_size], self.decoder_size, 
            stride=self.decoder_size)
        # scores: [B, num_classes, H, W]
        return scores
    
    def forward_decoder_linear(self, img, x):
        # x: [B, h, w, encoder_dim]
        h = x.shape[1]
        w = x.shape[2]

        x = torch.permute(x, [0, 3, 1, 2])
        # x: [B, encoder_dim, h, w]
        x = self.decoder(x)

        return x
        pass

    def forward_encoder(self, img):
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

        x = torch.reshape(x, [x.shape[0], h, w, x.shape[-1]])
        # x: [B, h, w, c]

        return x

    def forward(self, img):
        # img: [B, C, H, W]
        # x = self.forward_encoder(img)
        x1 = self.conv1(img)
        x1 = self.bn1(x1)
        x1 = self.multi_conv1(x1)
        x2 = self.conv2(F.relu(x1))
        # x2: [B, 256, h, w]
        x2 = self.bn2(x2)
        x3 = self.conv3(F.relu(x2))
        x3 = self.bn3(x3)
        x = torch.permute(x3, [0, 2, 3, 1])
        x = torch.reshape(x, [x.shape[0], -1, x.shape[-1]])
        pos_enc = self.encoder_position_encoding(self.encoder_size)
        x = self.encoder(x, structure_matrix=pos_enc)
        # x: [B, h*w, encoder_dim]
        x = torch.permute(x, [0, 2, 1])
        x = torch.reshape(x, [x.shape[0], x.shape[1], self.encoder_size, self.encoder_size])
        x = torch.cat([x, x3], axis=1)
        x = self.inv_conv1(x)
        x = self.inv_bn1(x)
        x = F.relu(x)
        x = torch.cat([x, x2], axis=1)
        x = self.inv_multi_conv1(x)
        x = self.inv_conv2(x)
        x = self.inv_bn2(x)
        x = F.relu(x)
        x = torch.cat([x, x1], axis=1)
        x = self.inv_multi_conv2(x)
        x = self.inv_conv3(x)
        x = self.inv_multi_conv3(x)
        x = torch.cat([x, img], axis=1)
        x = self.classifier(x)
        return x

        # return self.forward_decoder_transformer(img, x, h, w)
        # return self.forward_decoder_linear(img, x)
    
    def compute_loss(self, pixel_scores, mask):
        # pixel_scores: [B, num_classes, H, W]
        # mask: [B, H, W]
        loss = F.cross_entropy(pixel_scores, mask)
        return loss
    pass