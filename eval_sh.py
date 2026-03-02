# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from pathlib import Path
from typing import *

import slangtorch
import torch
from torch.autograd import Function
from icecream import ic

# Needed for normal lookup
from scipy.spatial import KDTree

kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "splinetracers/slang/sh_kernel.slang")
)

class EvalSH(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        features: torch.Tensor,
        rayo: torch.Tensor,
        sh_degree: int
    ):
        block_size = 64
        rayo = rayo.reshape(3).contiguous()
        means = means.contiguous()
        features = features.contiguous()
        color = torch.zeros_like(means)

        # XYZ Normal for each gaussian
        # (n x 3).
        normals = torch.zeros_like(means)
        # print(f"{normals.shape = }")

        generated_normals = EvalSH.make_normals(means, rayo)

        ctx.sh_degree = sh_degree
        num_prim = means.shape[0]
        kernels.sh_kernel(
            means=means, features=features, ray_origin=rayo, colors=color, sh_degree=sh_degree, normals=normals
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(num_prim // block_size + 1, 1, 1),
        )

        ctx.save_for_backward(
            means, features, rayo, color
        )
        return color
    
    @staticmethod
    def make_normals(means: torch.Tensor, ray_origin: torch.Tensor, k:int = 3) -> torch.Tensor:
        """
        Generate Normals from a point cloud (specified with means), with an additional ray_origin to orient normals towards.
        Uses PCA. Scheme inspired from https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html

        
        :param means: (n x 3) Tensor of locations for each point.
        :type means: torch.Tensor
        :param ray_origin: (3,) Tensor containing the viewpoint location of the normals.
        :type ray_origin: torch.Tensor
        :param k: Number of nearest-neighbors to consider around each point when calculating normals.
        :type k: int
        :return: (n x 3) Tensor of normals in object space that corresponds to each point.
        :rtype: Tensor
        """
        # Use local planarity assumption + PCA to get normals
        # TODO: A way to do this faster and take advantage of GPU?
        cpu_means = means.cpu()
        cpu_ray_origin = ray_origin.cpu()
        kd_tree = KDTree(cpu_means)

        # Find the k-nearest neighbors. We'll always get each point as it's closest nearest neighbor, so ask for one more.
        _, point_idxs = kd_tree.query(cpu_means, k=(k + 1))
        
        k_nearest_points = cpu_means[point_idxs] # (n, k + 1, 3)

        # Do PCA on k-nearest neighbors
        # https://docs.pytorch.org/docs/stable/generated/torch.pca_lowrank.html
        # Note that this is slightly non-deterministic.
        _, S, V = torch.pca_lowrank(k_nearest_points) # S has shape (n, 3). V has shape (n, 3, 3)

        # NOTE: V columns have PCA components.
        # We extract the first two principal components as the x and y axes of the plane, then use the cross product to get the z-axis:
        top_2_pcs = V[:, :, :2] # (n, 3, 2)
        x_pcs = V[:, :, 0] # (n x 3)
        y_pcs = V[:, :, 1] # (n x 3)

        normals = torch.linalg.cross(x_pcs, y_pcs, dim=1) # (n x 3)

        # Find whether to "flip" normals or not (normals should point towards camera):
        # Distance from point -> viewpoint
        point_to_cam_vecs = -(cpu_means - cpu_ray_origin.reshape(1, 3)) # (n x 3)

        # Ask if the point -> viewpoint vector the same sign as the normal vector
        normal_signs = torch.sign(torch.sum(point_to_cam_vecs * normals, dim = 1))
        normals = normals * normal_signs.reshape(-1, 1)

        return normals

    @staticmethod
    def backward(ctx, dL_dcolor: torch.Tensor):
        block_size = 64
        means, features, rayo, color = ctx.saved_tensors
        num_prim = means.shape[0]
        dL_dfeat = torch.zeros_like(features)
        dL_dcolor = dL_dcolor.contiguous()
        kernels.bw_sh_kernel(means=means, features=features, dL_dfeatures=dL_dfeat, ray_origin=rayo, dL_dcolors=dL_dcolor, sh_degree=ctx.sh_degree).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=((num_prim + block_size - 1) // block_size, 1, 1),
        )
        return None, dL_dfeat, None, None, None

def eval_sh(
        means,
        features,
        rayo,
        sh_degree):
    out = EvalSH.apply(
        means,
        features,
        rayo,
        sh_degree
    )
    return out

