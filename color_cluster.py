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
from connectivity.color_connectivity import ColorConnectGraph
from argparse import ArgumentParser
import argparse
import sys

def dfs(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node] and neighbor >= 0:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

def find_connected_components(graph):
    visited = set()
    components = []
    
    for node in graph:
        if node not in visited:
            component = set()
            dfs(graph, node, component)
            components.append(component)
            visited.update(component)
    
    return components

def get_color_clusters(ccgraph: ColorConnectGraph):
    
    visited = set()
    components = []
    
    for node in range(0, ccgraph):
        if node not in visited:
            component = set()
            dfs(ccgraph, node, component)
            components.append(component)
            visited.update(component)

    return components
    
def save_texture_segment_ply(inputpath, outputpath, rgb_truncate_threshold=15, intersect_threshold=1e-6, max_neighbor_num=10):

    intergraph = IntersectionGraph(max_neighbor_num=max_neighbor_num)
    intergraph.load_ply(inputpath)
    intergraph.get_connect_graph(threshold=intersect_threshold)

    color_connect = ColorConnectGraph()
    color_connect_graph = color_connect.get_scene_color_connect(intergraph, rgb_truncate=rgb_truncate_threshold)

    color_clusters = get_color_clusters(color_connect_graph)

    rgb_assign = torch.arange(0, 255*3, 255*3/len(color_clusters))
    rgb_label = torch.zeros((len(color_clusters), 3, 1))
    for idx in range(0, len(color_clusters)):

        if rgb_assign[idx] <= 255:
            rgb_label[idx] = [[rgb_assign[idx]], [0], [0]]
        else:
            if rgb_assign[idx] <= 255*2:
                rgb_label[idx] = [[255], [rgb_assign[idx] - 255], [0]]
            else:
                rgb_label[idx] = [[255], [255], [rgb_assign[idx] - 255*2]]

    segment_feature_dc = torch.zeros_like(intergraph._features_dc)

    cluster_idx=0
    for cluster in color_clusters:
        for gaussian_idx in cluster:
            segment_feature_dc[gaussian_idx] = rgb_label[cluster_idx]

        cluster_idx+=1
    
        xyz = intergraph._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = segment_feature_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() ##
        f_rest = intergraph._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = intergraph._opacity.detach().cpu().numpy()
        scale = intergraph._scaling.detach().cpu().numpy()
        rotation = intergraph._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in intergraph.construct_list_of_attributes()]

        ply_file_name = f"SpaceRGBThresh_{intersect_threshold}_{rgb_truncate_threshold}.ply"
        ply_file_path = os.path.join(outputpath, ply_file_name)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(ply_file_path)


if __name__ == "__main__":

    parser = ArgumentParser(description="segment script parameters")

    parser.add_argument("-input","--inputpath", type=str, default=None, help="path to 3D GS ply file")
    parser.add_argument("-output","--outputpath", type=str, default=None, help="path to where the segment result will output")
    parser.add_argument("--intersect_threshold", type=float, default=1e-6, help="metric for determining intersection")
    parser.add_argument("--RGB_threshold", type=int, default=15, help="metric for truncate rgb similarity")
    parser.add_argument("--max_neighbor_num", type=int, default=10, help="the max number of local neighbors")

    args = parser.parse_args(sys.argv[1:])

    if args.inputpath !=None and args.outputpath !=None :

        save_texture_segment_ply(args.inputpath, args.outputpath, args.RGB_threshold,args.intersect_threshold)

    else:
        if args.inputpath == None:
            print("invalid inputpath")
        else:
            print("invalid outputpath")
    
    


    
    