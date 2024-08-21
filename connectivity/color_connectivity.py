import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gaussian_model import GaussianModel
from connectivity.gaussian_intersection import IntersectionGraph
from tqdm import tqdm


class ColorConnectGraph:

    def __init__(self):
        self.color_connect_graph = torch.empty(0)

    def get_scene_color_connect(self, space_connect_graph: IntersectionGraph, rgb_truncate=15):
        
        color_connect = torch.full_like((space_connect_graph.K_neighbors), -2)

        for target_idx in tqdm(range(0, space_connect_graph.K_neighbors.shape[0]), desc="Processing of color connect"):
            true_neighbors=[neighbor for neighbor in space_connect_graph.K_neighbors[target_idx] if neighbor>=0]
            target_sh_dc = space_connect_graph._features_dc[target_idx]
            neighbor_sh_dc = space_connect_graph._features_dc[true_neighbors, :, :]

            sh_truncate = RGB2SH(rgb_truncate)

            sh_distribution = torch.abs(neighbor_sh_dc - target_sh_dc)
            sh_distances = torch.norm(sh_distribution, dim=1)
            
            color_neighbors = [true_neighbors[idx] for idx in range(0, len(true_neighbors)) if (sh_distances[idx] < torch.mean(sh_distances) and sh_distances[idx] < sh_truncate)]

            color_connect[target_idx][:len(color_neighbors)] = torch.tensor(color_neighbors)

        self.color_connect_graph = torch.tensor(color_connect)

        return color_connect





            