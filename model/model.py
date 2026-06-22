from torch.nn import Linear, Module, ModuleList, Parameter, functional as F
from .adapters import Adapter
import torch

class ZSADModel(Module):
    def __init__(self, args):
        super(ZSADModel, self).__init__()

        self.number_of_vision_layers = len(args.vision_layers)

        text_dim = args.text_dim
        vision_dim = args.vision_dim

        self.positive_text_adapter = Adapter(text_dim, args.out_dim, bottleneck=None, last_activation=True)
        self.negative_text_adapter = Adapter(text_dim, args.out_dim, bottleneck=None, last_activation=True)
        self.patch_adapter = ModuleList([Adapter(vision_dim, args.out_dim, bottleneck=None, last_activation=True)
                              for _ in range(self.number_of_vision_layers)])
        self.cls_adapter = ModuleList([Adapter(vision_dim, args.out_dim, bottleneck=None, last_activation=True)
                              for _ in range(self.number_of_vision_layers)])
        
        self.img_size = args.img_size

    def text_guided_anomaly_scoring(self, patch_features, adapted_text_features):
        B, N, _ = patch_features.shape
        grid_size = int(N**0.5)
        cross_model_text_guided = 100 * patch_features @ adapted_text_features
        cross_model_text_guided = F.interpolate(cross_model_text_guided.permute(0, 2, 1).view(-1, 2, grid_size, grid_size),
                                            size=self.img_size, mode='bilinear', align_corners=True)
        cross_model_text_guided = torch.softmax(cross_model_text_guided, dim=1)
        return cross_model_text_guided
    
    def global_context_regularizer(self, cls_features, adapted_text_features):
         return (100 * cls_features.unsqueeze(1) @ adapted_text_features).squeeze(1)
    
    def forward(self, text_embeddings, image_features):
        adapted_positive_text_embeddings = self.positive_text_adapter(text_embeddings['normal'])
        adapted_negative_text_embeddings = self.negative_text_adapter(text_embeddings['abnormal'])

        adapted_text_features = torch.stack([adapted_positive_text_embeddings, adapted_negative_text_embeddings], dim=-1)

        cls, patches = image_features

        text_guided_anomaly_maps = []
        global_context_regularizer_scores = []

        for i in range(4):
                cls_features = self.cls_adapter[i](cls[:,i,:])
                cls_features = cls_features / cls_features.norm(dim=-1, keepdim=True)
                patch_features = self.patch_adapter[i](patches[:,i,:,:])
                patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

                text_guided_anomaly_maps.append(self.text_guided_anomaly_scoring(patch_features, adapted_text_features))

                global_context_regularizer_scores.append(self.global_context_regularizer(cls_features, adapted_text_features.detach()))
            
        text_guided_anomaly_map = torch.mean(torch.stack(text_guided_anomaly_maps, dim=0), dim=0)
        global_context_regularizer_score = torch.mean(torch.stack(global_context_regularizer_scores, dim=0), dim=0)

        return text_guided_anomaly_map, global_context_regularizer_score




