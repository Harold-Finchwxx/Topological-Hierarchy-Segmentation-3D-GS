import torch
from torch import Tensor
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
from connectivity.color_connectivity import ColorConnectGraph
from argparse import ArgumentParser
import argparse
import sys
import time
from connectivity.color_cluster import dfs, get_clusters
import heapq


def get_scene_space_cluster(space_connect_graph_path=None, origin_gaussian_path=None, outputpath=None):

    if space_connect_graph_path == None:
        raise SystemExit("No space connect graph, please input one")
    if origin_gaussian_path == None:
        raise SystemExit("No origin gaussians, please input one")
    
    if not os.path.exists(space_connect_graph_path):
        raise SystemExit("Invalid space connect graph path")
    else:
        print("Valid space connect graph checked")

    if not os.path.exists(origin_gaussian_path):
        raise SystemExit("Invalid space connect graph path")
    else:
        print("Valid origin gaussian checked")

    space_connect_graph = torch.load(space_connect_graph_path)
    space_clusters = get_clusters(space_connect_graph)
    print(f"Clusters total number:{len(space_clusters)}")
    cluster_size = [len(cluster) for cluster in space_clusters]
    k = 10
    print(f"Size of Top {k} largest cluster:{heapq.nlargest(k, cluster_size)}")
    print(f"Size of the first cluster:{cluster_size[0]}")

    print('=' * 25 + 'Assigning Color Labels' + '=' * 25)

    rgb_label = torch.randint(0, 256, (len(space_clusters), 3, 1))
    rgb_label[0] = torch.tensor([[255], [0], [0]])
    sh_dc_label = RGB2SH(rgb_label)

    space_connect_feature_dc = torch.zeros((space_connect_graph.shape[0], 3, 1), dtype=torch.float)

    '''
    intergraph = IntersectionGraph()
    intergraph.load_ply(origin_gaussian_path)
    '''

    cluster_idx=0
    for cluster in space_clusters:
        for gaussian_idx in cluster:
            space_connect_feature_dc[gaussian_idx] = sh_dc_label[cluster_idx]

            # show origin render result of the first(probably the largest) cluster rather than assigning certain color
            '''
            if cluster_idx == 0:
                space_connect_feature_dc[gaussian_idx] = intergraph._features_dc[gaussian_idx]  
            '''
                
        cluster_idx+=1

    intergraph = IntersectionGraph()
    intergraph.load_ply(origin_gaussian_path)

    xyz = intergraph._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = torch.tensor(space_connect_feature_dc, dtype=torch.float).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() ##
    f_rest = torch.zeros_like(intergraph._features_rest).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = intergraph._opacity.detach().cpu().numpy()
    scale = intergraph._scaling.detach().cpu().numpy()
    rotation = intergraph._rotation.detach().cpu().numpy()
    
    print('=' * 25 + 'Saving Output File' + '=' * 25)

    dtype_full = [(attribute, 'f4') for attribute in intergraph.construct_list_of_attributes()]

    ply_file_name = f"SpaceConnectClusters_{time.time()}_MaxNeighbor_{space_connect_graph.shape[1]}.ply"
    ply_file_path = os.path.join(outputpath, ply_file_name)
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_file_path)

    print('=' * 25 + 'Output File Saved' + '=' * 25)



if __name__ == "__main__":

    parser = ArgumentParser(description="segment script parameters")

    parser.add_argument("-gaussian","--origin_gaussian_path", type=str, default=None, help="path to 3D GS ply file")
    parser.add_argument("-output","--outputpath", type=str, default=None, help="path to where the clusters result will output")
    parser.add_argument("-connect_graph","--space_connect_graph_path", type=str, default=None, help="path to the space connect graph")

    args = parser.parse_args(sys.argv[1:])

    if args.origin_gaussian_path !=None and args.outputpath !=None and args.space_connect_graph_path !=None:

        print('=' * 25 + 'Process Start' + '=' * 25)

        get_scene_space_cluster(args.space_connect_graph_path, args.origin_gaussian_path, args.outputpath)

    else:
        if args.origin_gaussian_path == None:
            print("No origin gaussian path")
        if args.outputpath == None:
            print("No outputpath")
        if args.space_connect_graph_path == None:
            print("No space connect graph")

        raise SystemExit()
    