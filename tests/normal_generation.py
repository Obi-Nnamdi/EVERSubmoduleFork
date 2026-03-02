import sys
sys.path.append(".")
from ever.eval_sh import EvalSH
import torch
import os
from pathlib import Path

# Graphing
import matplotlib
matplotlib.use('Agg') # headless mode
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with torch.no_grad():
        num_points = 40
        dim = 3
        k = 4
        box_size = 1
        box_points = 1000
        print(f"Testing Normal Creation ({k = }, {box_points = }):")

        test_means = torch.rand(num_points, 3)

        # Create a box point cloud to test our normal effectiveness:
        plane_points = torch.linspace(-box_size, box_size, box_points)

        box_volume_points = torch.cartesian_prod(plane_points, plane_points, plane_points) # (box_points**3 x 3)
        # Only keep points that have a coordinate on our boundary (abs_val == box_size) 
        box_boundary_points = box_volume_points[torch.any(box_volume_points.abs() == box_size, dim=1)] # (n x 3)

        # To reduce clutter, only plot positive octant points.
        front_facing_points = box_boundary_points[torch.all(box_boundary_points > 0, dim=1)]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')


        ax.scatter(front_facing_points[:, 0], front_facing_points[:, 1], front_facing_points[:, 2], marker="o", c=front_facing_points[:, 2], cmap='tab10')
        ax.set_title(f"Generated Normals ({k = }, {box_points = })")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.invert_yaxis()

        ray_origin = torch.tensor([2, 2, 2]).ravel()

        # Move inputs to CUDA for testing.
        # TODO: might be nice to use the openreg / meta backend for testing: https://github.com/pytorch/pytorch/issues/61654#issuecomment-879989145 and https://github.com/pytorch/pytorch/pull/155101#discussion_r2144157203
        device = torch.device("cuda")

        normals = EvalSH.make_normals(front_facing_points.to(device), ray_origin.to(device), k=k)
        normals = normals.cpu()
        # Plot output normals:
        # https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html
        ax.quiver3D(front_facing_points[:, 0], front_facing_points[:, 1], front_facing_points[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], length = 0.1, cmap='tab10', arrow_length_ratio=.4)

        save_path = Path(__file__).parent / "test_output.png"
        plt.savefig(save_path)
        print(f"Saved output figure at {str(save_path)}")