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
import tqdm


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
        K_neighbors = np.full((self._xyz.shape[0], self.max_neighbor_num), np.nan, dtype=int)
        max_scale = torch.exp(get_max_scale(self._scaling))
        scaling = torch.exp(self._scaling)

        for target_idx in tqdm(range(0, self._xyz.shape[0]), desc="Out Loop"):
            target_xyz = self._xyz[target_idx]
            space_truncate_mask = torch.all(torch.abs(self._xyz - target_xyz) < 2 * max_scale, dim=1)
            space_candidates = torch.nonzero(space_truncate_mask).squeeze().tolist()
            space_candidates.remove(target_idx)

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

        self.K_neighbors = torch.tensor(K_neighbors, dtype=int)
        return K_neighbors



        
                