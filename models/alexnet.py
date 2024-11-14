import torch
import torch.nn as nn
import torch.nn.functional as F
class AlexNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        #print(f"forward shape: {x.shape}")
        return x
    
# TODO add a custom distance func
# TODO Implement either Birch/Gaussian mixture/Spectral clustering, 

class KMeansClustering(nn.Module):
    def __init__(self, num_clusters=2, num_features=256, num_iter=10):
        super(KMeansClustering, self).__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.num_iter = num_iter
        self.centroids = nn.Parameter(torch.randn(num_clusters, num_features))

    def forward(self, x):
        for _ in range(self.num_iter):
            # Assign clusters
            distances = torch.cdist(x, self.centroids)
            assignments = torch.argmin(distances, dim=1)
            # Update centroids
            for k in range(self.num_clusters):
                cluster_points = x[assignments == k]
                if cluster_points.numel() > 0:
                    self.centroids.data[k] = cluster_points.mean(dim=0)
        return assignments
    

    