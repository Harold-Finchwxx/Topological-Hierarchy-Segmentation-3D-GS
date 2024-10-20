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
from tqdm import tqdm
import math

def get_max_scale(scales):
    return scales.max()

def build_covariance_from_scaling_rotation(scaling, rotation, scaling_modifier=1):
        if rotation.dim() == 1:
            rotation = rotation.unsqueeze(0)

        if scaling.dim() == 1:
            scaling = scaling.unsqueeze(0)

        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return actual_covariance


def compute_normalization_constant(mu1, Sigma1, mu2, Sigma2):
    # Convert inputs to torch tensors if they are not already
    mean1 = torch.tensor(mu1, dtype=torch.float32).clone().cuda()
    sigma1 = torch.tensor(Sigma1, dtype=torch.float32).clone().cuda()
    mean2 = torch.tensor(mu2, dtype=torch.float32).clone().cuda()
    sigma2 = torch.tensor(Sigma2, dtype=torch.float32).clone().cuda()

    # Compute the inverses of the covariance matrices
    inv_Sigma1 = torch.inverse(sigma1).cuda()
    inv_Sigma2 = torch.inverse(sigma2).cuda()
    
    # Compute the new covariance matrix (Sigma)
    Sigma = torch.inverse(inv_Sigma1 + inv_Sigma2).cuda()
    
    # Compute the mean difference vector
    m = mean1 - mean2
    
    # Compute the normalization constant
    dim = len(mean1)  # Dimension (3 in this case)
    normalization_constant = (2 * torch.pi)**(-dim / 2) * torch.sqrt(torch.det(Sigma)) * \
                             torch.exp(-0.5 * m.T @ torch.inverse(sigma1 + sigma2) @ m)
    
    return normalization_constant.item()



class IntersectionGraph:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree: int =3, max_neighbor_num: int =10):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.setup_functions()
        self.max_neighbor_num = max_neighbor_num
        self.K_neighbors = torch.empty(0)
        self._neighbors = []
        

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float)
        self._features_rest = torch.tensor(features_extra, dtype=torch.float)
        self._opacity = torch.tensor(opacities, dtype=torch.float)
        self._scaling = torch.tensor(scales, dtype=torch.float)
        self._rotation = torch.tensor(rots, dtype=torch.float)

        self.active_sh_degree = self.max_sh_degree

    def get_connect_graph(self, threshold=1e-6):
        # K_neighbors = np.full((self._xyz.shape[0], self.max_neighbor_num), np.nan, dtype=int)
        Neighbors = []
        max_scale = torch.exp(get_max_scale(self._scaling))
        scaling = torch.exp(self._scaling)

        # Get sorted coordinates
        x = self._xyz[:, 0].clone().squeeze().cuda()
        y = self._xyz[:, 1].clone().squeeze().cuda()
        z = self._xyz[:, 2].clone().squeeze().cuda()

        x_sorted, x_indices = torch.sort(x)
        y_sorted, y_indices = torch.sort(y)
        z_sorted, z_indices = torch.sort(z)

        x_sorted_with_origin_indices = torch.stack((x_sorted, x_indices), dim=1)
        y_sorted_with_origin_indices = torch.stack((y_sorted, y_indices), dim=1)
        z_sorted_with_origin_indices = torch.stack((z_sorted, z_indices), dim=1)

        for target_idx in tqdm(range(0, self._xyz.shape[0]), desc="Outter Loop of space intersection"):

            # Set the truncate boundary based on coordinate 
            truncate_distance_xyz = torch.full((3, ), max_scale * 1.75)
            target_xyz = self._xyz[target_idx].cuda()
            truncate_coordinate = torch.stack((target_xyz.squeeze() - truncate_distance_xyz, 
                                               target_xyz.squeeze() + truncate_distance_xyz), dim=0).cuda()
            
            x_bound_index = torch.searchsorted(x_sorted, truncate_coordinate[:, 0].squeeze())
            y_bound_index = torch.searchsorted(y_sorted, truncate_coordinate[:, 1].squeeze())
            z_bound_index = torch.searchsorted(z_sorted, truncate_coordinate[:, 2].squeeze())

            # Get original indices for truncated candiadates
            x_candidates = x_sorted_with_origin_indices[x_bound_index[0]: x_bound_index[1], 1].squeeze()
            y_candidates = y_sorted_with_origin_indices[y_bound_index[0]: y_bound_index[1], 1].squeeze()
            z_candidates = z_sorted_with_origin_indices[z_bound_index[0]: z_bound_index[1], 1].squeeze()

            xyz_candidates = torch.cat((x_candidates, y_candidates, z_candidates), dim=0).cuda()
            unique_elements, counts = torch.unique(xyz_candidates, return_counts=True)
            candidates_id = unique_elements[counts == 3]

            # Secondary filter by space distance
            distance_to_target = torch.norm(self._xyz[candidates_id, :] - target_xyz, dim=1, keepdim=False)
            neighbors_mask = (distance_to_target < 2 * scaling[target_idx].max())
            neighbors = torch.tensor(candidates_id[neighbors_mask], dtype=int).squeeze().tolist()
            neighbors.remove(target_idx)

            Neighbors.append(neighbors)

            '''
            target_xyz = self._xyz[target_idx]
            distance_to_target = torch.norm(self._xyz - target_xyz, dim=1, keepdim=False)
            space_truncate_mask = (distance_to_target <= 2 * torch.exp(self._scaling[target_idx]).max())
            space_candidates = torch.nonzero(space_truncate_mask).squeeze().tolist()
            if isinstance(space_candidates, int):
                space_candidates = [space_candidates]
            space_candidates.remove(target_idx)
            '''

            '''
            target_sigma = build_covariance_from_scaling_rotation(scaling[target_idx], self._rotation[target_idx])
            intersec_candidates = []

            for candidate in tqdm(space_candidates, desc="Inner Loop"):
                canditate_xyz = self._xyz[candidate]
                candidate_sigma = build_covariance_from_scaling_rotation(scaling[candidate], self._rotation[candidate])
                intersect_metric = compute_normalization_constant(target_xyz, target_sigma, canditate_xyz, candidate_sigma)
                if intersect_metric > threshold:
                    intersec_candidates.append([candidate, intersect_metric])

            if len(intersec_candidates) <= self.max_neighbor_num:
                K_neighbors[target_idx][:len(intersec_candidates)] = [c[0] for c in intersec_candidates]
            else:
                intersec_candidates.sort(key=lambda x: x[1], reverse=True)
                K_neighbors[target_idx] = [c[0] for c in intersec_candidates[:self.max_neighbor_num]]
            '''

            '''
            if len(space_candidates) <= self.max_neighbor_num:
                K_neighbors[target_idx][:len(space_candidates)] = [c for c in space_candidates]
            else:
                space_candidates.sort(key=lambda x: distance_to_target[x])
                K_neighbors[target_idx] = [c for c in space_candidates[:self.max_neighbor_num]]
            '''

        # self.K_neighbors = torch.tensor(K_neighbors, dtype=int)
        # return K_neighbors
        self._neighbors = Neighbors
        return Neighbors



        
                