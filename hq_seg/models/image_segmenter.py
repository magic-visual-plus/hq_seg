import torch.nn as nn
import torch.nn.functional as F
import torch
from . import simple_transformer
from torchvision.ops import sigmoid_focal_loss
from . import cross_transformer


class PositionEncoding1DEx(nn.Module):
    def __init__(self, query_max_size, key_max_size, dim):
        super(PositionEncoding1DEx, self).__init__()
        self.dim = dim
        self.x_emb = nn.Embedding(query_max_size, dim)
        self.y_emb = nn.Embedding(key_max_size, dim)
        pass

    def forward(self, query_size, key_size):
        x = torch.arange(query_size, device=self.x_emb.weight.device)
        y = torch.arange(key_size, device=self.y_emb.weight.device)
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 0)
        x = torch.tile(x, [1, key_size])
        y = torch.tile(y, [query_size, 1])
        x_encoding = self.x_emb(x)
        y_encoding = self.y_emb(y)
        return x_encoding + y_encoding
        pass


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

class ImageEncoderTransformer(nn.Module):
    def __init__(
            self, kernel_size, stride, num_layers1, hidden_size, output_size, num_layers2):
        super(ImageEncoderTransformer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers1 = num_layers1
        self.hidden_size = hidden_size
        self.pixel_embed_layer = nn.Linear(3, hidden_size)
        self.attention_head_size = hidden_size // 4
        self.attention_head_size2 = output_size // 8
        self.encode_layer = simple_transformer.MultiLayerTransformer(
            num_layers1, hidden_size, attention_head_size=self.attention_head_size,
            reduction=None, use_structure_matrix=True)
        self.position_encoder = PositionEncoding2D(kernel_size, self.attention_head_size)

        self.output_layer1 = nn.Linear(hidden_size, output_size)
        self.output_position_encoder = PositionEncoding1DEx(kernel_size**2, output_size, self.attention_head_size2)
        self.output_layer = cross_transformer.MultiLayerCrossTransformer(
            num_layers=1, hidden_size=output_size, attention_head_size=self.attention_head_size2,
            reduction=None, use_structure_matrix=True
        )
        # self.output_layer = nn.Linear(self.hidden_size * self.kernel_size**2, output_size)

        self.encode_layer2 = simple_transformer.MultiLayerTransformer(
            num_layers2, output_size, attention_head_size=self.attention_head_size2,
            reduction=None, use_structure_matrix=True)
        self.position_encoder2 = PositionEncoding2D(kernel_size**2, self.attention_head_size2)
        pass

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == W
        L = H * W // self.stride**2
        x = F.unfold(x, self.kernel_size, stride=self.stride)
        # x: [B, C*kernel_size**2, H*W//stride**2]
        x = x.reshape(B, C, self.kernel_size**2, L)
        x = x.permute(0, 3, 2, 1)
        # x: [B, L, kernel_size**2, C]
        x = self.pixel_embed_layer(x)
        # x: [B, L, kernel_size**2, h]
        x = x.reshape(-1, self.kernel_size**2, self.hidden_size)
        pos_code = self.position_encoder(self.kernel_size)
        # x: [B*L, kernel_size**2, h]
        x = self.encode_layer(x, structure_matrix=pos_code)
        # x: [B*L, kernel_size**2, h]

        x = self.output_layer1(x)
        # x: [B*L, kernel_size**2, output_size]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        # x: [B*L, 1, output_size]
        
        pos_code_output = self.output_position_encoder(1, self.kernel_size**2)
        x = self.output_layer(
            x_mean, x, structure_matrix=pos_code_output)
        # x: [B*L, 1, output_size]
        x = x.reshape(B, L, -1)
        # x: [B, L, output_size]

        # x = x.reshape(B, L, -1)
        # # x: [B, L, kernel_size**2*h]
        # x = self.output_layer(x)
        # # x: [B, L, output_size]
        
        w = W // self.stride
        pos_code2 = self.position_encoder2(w)
        x = self.encode_layer2(x, structure_matrix=pos_code2)
        # x: [B, L, output_size]

        x = x.reshape(B, w, w, -1)

        return x



class PixelDecoderTransformer(nn.Module):
    def __init__(self, num_classes, input_size, kernel_size, stride, num_layers, hidden_size):
        super(PixelDecoderTransformer, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.stride = stride
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // 4
        self.attention_head_size2 = input_size // 8

        self.position_encoder = PositionEncoding2D(self.kernel_size, self.attention_head_size)

        self.transformer = simple_transformer.MultiLayerTransformer(
            num_layers, hidden_size, attention_head_size=self.attention_head_size, reduction=None, use_structure_matrix=True)
        
        self.linear_map = nn.Linear(input_size, hidden_size)
        self.cross_map = cross_transformer.MultiLayerCrossTransformer(
            num_layers=1, hidden_size=input_size, attention_head_size=self.attention_head_size2,
            reduction=None, use_structure_matrix=True)
        self.position_encoder_map = PositionEncoding1DEx(kernel_size**2, 1, self.attention_head_size2)

        self.pixel_embed_layer = nn.Linear(3, input_size)
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        pass

    def forward(self, embed, x):
        # embed: [B, w, w, input_size]
        # x: [B, C, H, W]
        H = x.shape[2]
        W = x.shape[3]
        # embed = embed.permute(0, 3, 1, 2)
        # # embed: [B, input_size, w, w]
        # embed = self.inv_conv(embed)
        # # embed: [B, hidden_size, H, W]
        # embed = embed.permute(0, 2, 3, 1)

        B = embed.shape[0]
        embed = embed.reshape((-1, 1, self.input_size))
        # embed: [B*w*w, 1, hidden_size]
        pos_code_map = self.position_encoder_map(self.kernel_size**2, 1)
        
        x = x.permute(0, 2, 3, 1)
        # x: [B, H, W, C]
        x = self.pixel_embed_layer(x)
        L = x.shape[1] * x.shape[2] // self.stride**2

        # x: [B, H, W, input_size]
        x = F.unfold(x, self.kernel_size, stride=self.stride)
        # x: [B, input_size*kernel_size**2, L]
        x = x.reshape((B, self.input_size, self.kernel_size**2, -1))
        # x: [B, hidden_size, kernel_size**2, L]
        x = x.permute(0, 3, 2, 1)
        # x: [B, L, kernel_size**2, hidden_size]
        x = x.reshape((-1, self.kernel_size**2, self.input_size))
        # x: [B*L, kernel_size**2, hidden_size]

        x = self.cross_map(x, embed, structure_matrix=pos_code_map)
        # x: [B*L, kernel_size**2, hidden_size]

        x = self.linear_map(x)
        # x = x + embed
        # B = x.shape[0]
        # x = x.permute(0, 3, 1, 2)
        # # x: [B, hidden_size, H, W]
        # x = F.unfold(x, self.kernel_size, stride=self.stride)
        # # x: [B, hidden_size*kernel_size**2, L]
        # x = x.reshape((B, self.hidden_size, self.kernel_size**2, L))
        # x = x.permute(0, 3, 2, 1)
        # # x: [B, L, kernel_size**2, hidden_size]
        # x = x.reshape((-1, self.kernel_size**2, self.hidden_size))
        # x: [B*L, kernel_size**2, hidden_size]
        pos_code = self.position_encoder(self.kernel_size)
        x = self.transformer(x, structure_matrix=pos_code)
        # x: [B*L, kernel_size**2, hidden_size]
        x = self.classifier(x)
        # x: [B*L, kernel_size**2, num_classes]
        x = x.reshape(B, L, self.kernel_size**2, -1)
        # x: [B, L, kernel_size**2, num_classes]
        x = x.permute(0, 3, 2, 1)
        # x: [B, num_classes, kernel_size**2, L]
        x = x.reshape(B, -1, L)
        # x: [B, num_classes*kernel_size**2, L]
        x = F.fold(x, [H, W], self.kernel_size, stride=self.stride)

        return x
        pass


class ImageSegmenter2(nn.Module):
    def __init__(self, num_classes, encoder_size, hidden_size, decoder_size, kernel_size, stride, num_layers_encoder, num_layers_hidden, num_layers_decoder):
        super(ImageSegmenter2, self).__init__()
        self.encoder = ImageEncoderTransformer(
            kernel_size, stride, num_layers_encoder, encoder_size, hidden_size, num_layers_hidden)
        self.decoder = PixelDecoderTransformer(
            num_classes, hidden_size, kernel_size, stride, num_layers_decoder, decoder_size)
        pass

    def forward(self, x):
        emb = self.encoder(x)
        x = self.decoder(emb, x)
        return x
        pass

    def compute_loss(self, pixel_scores, mask):
        # pixel_scores: [B, num_classes, H, W]
        # mask: [B, H, W]
        
        # focal loss
        proba = F.softmax(pixel_scores, dim=1)
        # proba: [B, num_classes, H, W
        loss = F.cross_entropy(pixel_scores, mask, reduction='none')
        # loss: [B, H, W]
        proba = torch.gather(proba, 1, torch.unsqueeze(mask, 1)).detach()

        loss = torch.where(proba > 0.9, 0, loss)
        loss = torch.mean(loss)
        return loss

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
        self.conv1 = nn.Conv2d(3, 128, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.multi_conv1 = MultiConv2d(128, 128, 3, 1, 1)

        encoder_num_heads = 16
        encoder_head_size = encoder_dim // encoder_num_heads
        self.encoder = simple_transformer.MultiLayerTransformer(
            6, encoder_dim, encoder_dim*4, encoder_head_size, encoder_num_heads, 0.1, 0.1, None, True)
        # self.decoder = ImageSegmenterDecoderTransformer(num_classes, decoder_dim, decoder_size)
        self.encoder_position_encoding = PositionEncoding2D(self.encoder_size, encoder_head_size)

        self.classifer_linear = nn.Linear(encoder_dim, decoder_size*decoder_size*num_classes)
        self.inv_conv1 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2, padding=0)
        self.inv_conv2 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=0)
        self.inv_conv3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0)
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
        
        # focal loss
        proba = F.softmax(pixel_scores, dim=1)
        # proba: [B, num_classes, H, W
        loss = F.cross_entropy(pixel_scores, mask, reduction='none')
        # loss: [B, H, W]
        proba = torch.gather(proba, 1, torch.unsqueeze(mask, 1)).detach()

        loss = torch.where(proba > 0.9, 0, loss)
        loss = torch.mean(loss)
        return loss
    pass