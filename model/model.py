# implementation from https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py

import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        # gives us a "soft assignment" of each descriptor to the clusters. 
        # bias added to output of layer which is a trainable parameter, meaning value is learned during training and allows neural network to fit data better. 
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        # torch.rand(num_clusters, dim) used to initialize centroids with random values. generates a tensor of the given shape 1 filled with random numbers from a uniform distribution on interval 0 to 1.
        # centroids variable is included as model parameter that would be optimized during training
        # torch.rand generates random value tensor num_clusters as rows and dim as cols
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        # calls function defined below
        self._init_params()

    # initializes the weights and biases of the convolutional layer based on the centroids of the clusters
    # By setting the weights of a convolutional layer to the centroids, each convolution operation can be thought of as measuring the similarity between the input descriptors and the centroids. 
    # The closer a descriptor is to a centroid, the higher the response (output value) of the convolution operation for that centroid's filter.
    def _init_params(self):
        # Multiplying the centroids by 2.0 * self.alpha is a way to initialize the scale of the weights. The factor of 2.0 is likely chosen based on empirical results
        # unsqueeze(-1).unsqueeze(-1) part is used to add two extra dimensions to the end of the tensor, which is required because the weights of a Conv2d layer in PyTorch need to have four dimensions (out_channels, in_channels, kernel_height, kernel_width)
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        # norm(dim=1) part is calculating the L2 norm (Euclidean length) of the centroids along dimension 1 (the descriptor dimension), and this is then multiplied by -self.alpha
        # the bias is initialized to be the negative norm of the centroids scaled by self.alpha. 
        # negative norm by self.alpha scales the effect that each centroid's magnitude has on the initial soft-assignment
        # simply, its measuring how far away each centroid is from the origin (0 point) in terms of its features, then flipping that distance to point backwards and scaling it up or down. 
        # so the negative alpha is pushing the decision boundary away from centroid. means that a descriptor has to be closer to a centroid to be assigned to its cluster and prevents descriptors that are far away from any centroid from being ambiguously assigned
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        # grabs shape of input tensor. N = batch size, C = num channels in each image
        N, C = x.shape[:2]

        # if normalize=True, then normalize along descriptor dimension. p=2 stands for L2 norm or euclidean norm
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        # reshaped to have N rows (for each image) and self.num_clusters columns (one for each cluster), and the rest in the last dimension. -1 calcs size of last dimension (total num of elements in each elements after conv operation)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # softmax applied applied to the descriptors to turn them into probabilities that sum to 1
        soft_assign = F.softmax(soft_assign, dim=1)

        # flattened into two dimensions (batch size and channels). 1 calcs output size
        x_flatten = x.view(N, C, -1)
        
        # calculate the residuals (differences) between each descriptor and each cluster centroid
        # permute changes order of dimensions of a tensor by index
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # weighted by the soft assignments
        residual *= soft_assign.unsqueeze(2)
        # These residuals are summed up to get the VLAD (Vector of Locally Aggregated Descriptors) for each image. Infers size of this dimension.
        vlad = residual.sum(dim=-1)

        # intra-normalization, VLADs are normalized within each descriptor. making sure that the length (or magnitude) of each descriptor within the VLAD vectors is 1, according to the L2 norm. 
        vlad = F.normalize(vlad, p=2, dim=2)  
        # flattened into 1 demension, output inferred
        vlad = vlad.view(x.size(0), -1) 
        # The flattened VLADs are L2 normalized.
        vlad = F.normalize(vlad, p=2, dim=1) 

        # returns the VLADs, which are the output of the NetVLAD layer. These can be used as global descriptors of the images for retrieval tasks.
        return vlad
