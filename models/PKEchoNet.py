import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torchvision import models


class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_head=32, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.msa = nn.MultiheadAttention(dim_in, dim_in // dim_head,
                                         dropout_rate, batch_first=True)
        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(dim_in)        

    def forward(self, x):
        msa_x = self.msa(x, x, x)[0]
        x = x + self.drop(msa_x)
        x = self.norm(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_rate=4, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_in, dim_in * dim_rate),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_in * dim_rate, dim_in)
        )
        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(dim_in)
    
    def forward(self, x):
        ffn_x = self.ffn(x)
        x = x + self.drop(ffn_x)
        x = self.norm(x)

        return x


class AffinityEstimator(nn.Module):
    def __init__(self, in_channels, num_nebs=9, scale_rate=2):
        super(AffinityEstimator, self).__init__()
        self.num_nebs = num_nebs

        self.estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_nebs * scale_rate**2, kernel_size=3, padding=1)
        ) 

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1) 
    
    def forward(self, features):
        nebs = self.num_nebs

        affinities = self.estimator(features) # B*T,nebs*hw,H,W
        affinities = rearrange(affinities, "BT (nebs hw) H W -> BT H W hw nebs", nebs=nebs)

        b = int(nebs**0.5)
        
        # handle borders
        affinities[:,0,:,:,:b] = float('-inf')
        affinities[:,-1,:,:,-b:] = float('-inf') 
        affinities[:,:,0,:,::b] = float('-inf')
        affinities[:,:,-1,:,b-1::b] = float('-inf')

        affinities = affinities.softmax(dim=-1)

        return affinities


class HybridAttention(nn.Module):
    def __init__(self, dim_in, dim_head=32, dropout_rate=0.1):
        super(HybridAttention, self).__init__()
        self.feature_msa = nn.MultiheadAttention(dim_in, dim_in // dim_head, 
                                                 dropout_rate, batch_first=True)
        self.kernel_msa = nn.MultiheadAttention(dim_in, dim_in // dim_head, 
                                                dropout_rate, batch_first=True)

        self.feature_drop = nn.Dropout(dropout_rate)
        self.kernel_drop = nn.Dropout(dropout_rate)

        self.feature_norm = nn.LayerNorm(dim_in)
        self.kernel_norm = nn.LayerNorm(dim_in)

        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LayerNorm(dim_in),
            nn.ReLU()
        )

    def forward(self, features, kernels):
        mas_features = self.feature_msa(kernels, features, features)[0]
        mas_kernels = self.kernel_msa(features, kernels, kernels)[0]

        features = kernels + self.feature_drop(mas_features)
        kernels = features + self.kernel_drop(mas_kernels)

        hybrid = self.feature_norm(features) + self.kernel_norm(kernels)
        hybrid = self.fc(hybrid)

        return hybrid


class KSS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dim_head=32, dim_rate=4, 
                 num_stage=2, num_frame=10, num_classes=1, dropout_rate=0.1):
        super(KSS, self).__init__() 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_stage = num_stage
        self.num_frame = num_frame
        self.num_classes = num_classes
        
        self.predictor_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.predictor_conv2 = nn.Conv2d(out_channels, num_classes, 
                                         kernel_size=kernel_size, 
                                         padding=(kernel_size-1)//2)

        self.pos_embed = nn.Parameter(torch.randn(1, num_frame, out_channels))

        self.updator = nn.ModuleList([])
        self.bias = nn.ParameterList()
        for _ in range(num_stage):
            self.updator.append(nn.ModuleList([
                HybridAttention(out_channels, dim_head, dropout_rate),
                SelfAttention(out_channels, dim_head, dropout_rate),
                SelfAttention(out_channels, dim_head, dropout_rate),
                FeedForward(out_channels, dim_rate, dropout_rate),
            ])) 
            self.bias.append(nn.Parameter(torch.zeros(num_classes)))
            
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1) 

    def forward(self, features):
        C = self.out_channels
        K = self.kernel_size
        T = self.num_frame
        B = features.size(0) // T
        ncls = self.num_classes
        
        # initial mask
        features = self.predictor_conv1(features)
        masks = self.predictor_conv2(features)

        predict_masks = [masks]
        
        # initial kernels
        kernels = self.predictor_conv2.weight.clone()
        kernels = kernels[None].expand(B*T, *kernels.size())
        kernels = kernels.view(B,T,*kernels.shape[1:])

        for i, updator in enumerate(self.updator):
            masks = masks.sigmoid() 

            # grouping features by mask
            group_features = torch.einsum('bnhw,bchw->bnc', masks, features) 
            group_features = group_features.view(-1,1,C).repeat(1,K**2,1) # B*T*ncls,1,C
            
            # hybrid attention
            kernels = kernels.view(B*T*ncls,C,-1).transpose(-1,-2) # B*T*ncls,K**2,C
            kernels = updator[0](group_features, kernels)

            # temporal attention
            kernels = rearrange(kernels, '(b t ncls) n c -> (b n ncls) t c', b=B, ncls=ncls) 
            if i == 0:
                kernels += self.pos_embed 
            kernels = updator[1](kernels)

            # cross-video attention
            kernels = rearrange(kernels, '(b n ncls) t c -> (t n ncls) b c', b=B, ncls=ncls) 
            kernels = updator[2](kernels)

            # feed-forward network
            kernels = rearrange(kernels, '(t n ncls) b c -> (b t ncls) n c', t=T, ncls=ncls) 
            kernels = updator[3](kernels)
    
            kernels = kernels.view(B*T,ncls,K,K,C).permute(0,1,4,2,3)

            # predicting new masks
            masks = []
            for j in range(B*T):
                masks.append(F.conv2d(features[j:j+1], kernels[j], self.bias[i], padding=(K-1)//2))
            masks = torch.cat(masks, dim=0) # B*T,ncls,H,W

            predict_masks.append(masks)
        
        return predict_masks


class PKEchoNet(nn.Module):
    def __init__(self, in_channels=1, neb_range=3, kernel_size=1, 
                 dim_head=32, dim_rate=4, num_stage=3, num_frame=10,
                 num_classes=1, dropout_rate=0.1, backbone='resnet50'):
        super(PKEchoNet, self).__init__()

        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)  
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)  
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError("Expected backbone is resnet18/34/50!")

        if in_channels == 3:
            conv1 = resnet.conv1
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")

        self.encoder_layer0 = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4

        num_nebs = neb_range ** 2

        self.estimator1 = AffinityEstimator(resnet.layer1[-1].conv1.in_channels, 
                                            num_nebs=num_nebs, scale_rate=4)
        self.estimator2 = AffinityEstimator(resnet.layer2[-1].conv1.in_channels, 
                                            num_nebs=num_nebs, scale_rate=2)
        self.estimator3 = AffinityEstimator(resnet.layer3[-1].conv1.in_channels, 
                                            num_nebs=num_nebs, scale_rate=2)
        self.estimator4 = AffinityEstimator(resnet.layer4[-1].conv1.in_channels, 
                                            num_nebs=num_nebs, scale_rate=2)

        self.kss = KSS(resnet.layer4[-1].conv1.in_channels, 32, kernel_size=kernel_size, 
                       dim_head=dim_head, dim_rate=dim_rate, num_stage=num_stage, 
                       num_frame=num_frame, num_classes=num_classes, dropout_rate=dropout_rate)

        self.unfold = nn.Unfold(kernel_size=neb_range, padding=(neb_range-1)//2)

    def upsampling(self, x, m):
        H = x.size(-2)
        hw, nebs = m.shape[-2:]

        x = self.unfold(x) # B*T,ncls*nebs,H*W
        x = rearrange(x, 'BT (ncls nebs) (H W) -> BT H W nebs ncls', H=H, nebs=nebs)

        x = torch.einsum('bhwpn,bhwnc->bhwpc', m, x) # B*T,H,W,h*w,ncls
        x = rearrange(x, 'BT H W (h w) ncls -> BT ncls (H h) (W w)', h=int(hw**0.5))

        return x
    
    def forward(self, x):
        B, T, C, H, W = x.size()

        x = x.view(-1, C, H, W)
    
        x = self.encoder_layer0(x)
        
        multi_affinities = []

        x = self.encoder_layer1(x)
        multi_affinities.insert(0, self.estimator1(x))

        x = self.encoder_layer2(x)
        multi_affinities.insert(0, self.estimator2(x))

        x = self.encoder_layer3(x)
        multi_affinities.insert(0, self.estimator3(x))

        x = self.encoder_layer4(x)
        multi_affinities.insert(0, self.estimator4(x))

        predict_masks = self.kss(x)

        for i in range(len(multi_affinities)):
            for j in range(len(predict_masks)):
                predict_masks[j] = self.upsampling(predict_masks[j], multi_affinities[i])

        for i in range(len(predict_masks)):
            masks = predict_masks[i]
            masks = masks.view(B, T, -1, H, W)
            if self.training:
                masks = [masks[:,:1], masks[:,-1:]]
                masks = torch.cat(masks, dim=1)
            predict_masks[i] = masks
     
        return predict_masks
    

if __name__ == '__main__':
    input = torch.rand(1,10,1,320,320).cuda()
    model = PKEchoNet().cuda()
    model(input)