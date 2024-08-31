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
from tqdm import tqdm

'''
def dfs(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node, :]:
        if neighbor>=0 and neighbor not in visited:
            dfs(graph, int(neighbor.item()), visited)
'''

def dfs(graph, start_node, cluster_visited, scene_visited, pbar):
    stack = [start_node]  # initiate stack, including start node
    while stack:
        node = stack.pop()  # pop top element of the pop
        if (node not in cluster_visited) and (node not in scene_visited):
            cluster_visited.add(int(node))  # mark current node as visited
            scene_visited.add(int(node))
            pbar.update(1)
            for neighbor in graph[node, :]:  # go through all neighbors of current node
                true_neighbor = int(neighbor.item())
                if (true_neighbor >= 0) and (true_neighbor not in cluster_visited) and (true_neighbor not in scene_visited):
                    stack.append(int(true_neighbor))  # push unvisited neighbors into the stack


'''
def find_connected_components(graph) -> list:
    visited = set()
    components = []
    
    for node in graph:
        if node not in visited:
            component = set()
            dfs(graph, node, component)
            components.append(component)
            visited.update(component)
    
    return components
'''

def get_clusters(ccgraph: Tensor) -> list:
    
    visited = set()
    components = []

    print('=' * 25 + 'Geting Clusters' + '=' *25)
    
    processingbar = tqdm(total=ccgraph.shape[0], desc="Clustering Process")

    for node in range(0, ccgraph.shape[0]):
        if node not in visited:
            component = set()
            dfs(ccgraph, node, component, visited, processingbar)
            components.append(component)
            visited.update(component)
            # processingbar.update(len(component))

    processingbar.close()
    print('=' * 25 + 'Clusters Got' + '=' *25)

    return components
    
def save_texture_segment_ply(inputpath, outputpath, rgb_truncate_threshold=15, intersect_threshold=1e-6, max_neighbor_num=10):
    
    print("=" * 25 + "Testing Input and Ouput Path" + "=" *25)

    if os.path.exists(inputpath):
        print("Valid input path")
    else:
        print("Invalid input path")
        raise SystemExit()
    
    if os.path.exists(outputpath):
        print("Valid output path")
    else:
        print("output path does not exist")
        os.makedirs(outputpath)
        print("But a new output diractory is established of this")
    
    print("=" * 25 + "Input and Ouput Path Tested" + "=" *25)

    print('\n')

    print("=" * 25 + "Establishing Space Connectivity Graph" + "=" *25)

    intergraph = IntersectionGraph(max_neighbor_num=max_neighbor_num)
    intergraph.load_ply(inputpath)
    intergraph.get_connect_graph(threshold=intersect_threshold)
    neighbor_tensor_filename = f"space_neighbors_MaxNeighbor_{max_neighbor_num}.pt"
    space_neighbor_tensor_path = os.path.join(outputpath, neighbor_tensor_filename)
    torch.save(intergraph.K_neighbors,space_neighbor_tensor_path)

    print("=" * 25 + "Space Connectivity Graph Saved" + "=" *25)

    print('\n')

    print("=" * 25 + "Establishing Color Connectivity Graph" + "=" *25)

    color_connect = ColorConnectGraph()
    color_connect_graph = color_connect.get_scene_color_connect(intergraph, rgb_truncate=rgb_truncate_threshold)
    color_connect_filename = f"color_neighbors_MaxNeighbor_{max_neighbor_num}_ColorThreshold_{rgb_truncate_threshold}.pt"
    color_connect_tensor_path = os.path.join(outputpath, color_connect_filename)
    torch.save(color_connect_graph, color_connect_tensor_path)

    print("=" * 25 + "Color Connectivity Graph Saved" + "=" *25)

    print('\n')

    print("=" * 25 + "Establishing Color Clusters Graph" + "=" *25)

    color_clusters = get_clusters(color_connect_graph)
    color_cluster_filename = "color_clusters.pt"
    color_cluster_list_path = os.path.join(outputpath, color_cluster_filename)
    torch.save(color_clusters, color_cluster_list_path)

    '''
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
    '''

    rgb_label = torch.randint(0, 256, (len(color_clusters), 3, 1))
    sh_dc_label = RGB2SH(rgb_label)

    segment_feature_dc = torch.zeros_like(intergraph._features_dc)

    cluster_idx=0
    for cluster in color_clusters:
        for gaussian_idx in cluster:
            segment_feature_dc[gaussian_idx] = sh_dc_label[cluster_idx]

        cluster_idx+=1
    
    xyz = intergraph._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = segment_feature_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() ##
    f_rest = intergraph._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = intergraph._opacity.detach().cpu().numpy()
    scale = intergraph._scaling.detach().cpu().numpy()
    rotation = intergraph._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in intergraph.construct_list_of_attributes()]

    ply_file_name = f"SpaceRGBThresh_{intersect_threshold}_{rgb_truncate_threshold}_maxneighbor_{max_neighbor_num}.ply"
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

        save_texture_segment_ply(args.inputpath, args.outputpath, args.RGB_threshold ,args.intersect_threshold, args.max_neighbor_num)

    else:
        if args.inputpath == None:
            print("No inputpath")
        else:
            print("No outputpath")
    
        raise SystemExit()
    