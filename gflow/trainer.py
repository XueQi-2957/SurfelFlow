import math
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import imageio
import msplat
import utils
from datetime import datetime
import os
import shutil
import time
from pathlib import Path
import utils.render as render
import utils.render_2dgs as render_2dgs
import cv2
import utils.geometry as geometry
from sklearn.cluster import KMeans
import roma
try:
    from render_policy import should_render_split_diagnostics, train_render_return_types
except ImportError:
    from .render_policy import should_render_split_diagnostics, train_render_return_types


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _sample_mask_weights(mask_tensor, uv, width, height):
    """Sample a single-channel mask tensor at arbitrary UV coordinates."""
    if mask_tensor is None or uv.numel() == 0:
        return uv.new_zeros((uv.shape[0],), dtype=uv.dtype)
    grid = uv.clone()
    grid[:, 0] = 2.0 * (grid[:, 0] / max(width - 1, 1)) - 1.0
    grid[:, 1] = 2.0 * (grid[:, 1] / max(height - 1, 1)) - 1.0
    grid = grid.clamp(-1.0, 1.0).view(1, -1, 1, 2)
    return torch.nn.functional.grid_sample(
        mask_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)


def _compute_footprint_overlap_ratio(mask_tensor, centers, radii, conics, width, height):
    """Approximate stale-mask overlap ratio over a 2DGS footprint using conic-weighted samples."""
    if (
        mask_tensor is None
        or centers.numel() == 0
        or radii is None
        or conics is None
    ):
        return centers.new_zeros((centers.shape[0],), dtype=centers.dtype)

    device = centers.device
    dtype = centers.dtype
    # 3x3 samples over the main footprint region; conic weights provide the soft area ratio.
    sample_lattice = torch.tensor(
        [
            [-0.5, -0.5],
            [0.0, -0.5],
            [0.5, -0.5],
            [-0.5, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [-0.5, 0.5],
            [0.0, 0.5],
            [0.5, 0.5],
        ],
        device=device,
        dtype=dtype,
    )
    sample_offsets = radii.unsqueeze(1) * sample_lattice.unsqueeze(0)  # (N, 9, 2)
    sample_uv = centers.unsqueeze(1) + sample_offsets  # (N, 9, 2)
    mask_samples = _sample_mask_weights(
        mask_tensor,
        sample_uv.reshape(-1, 2),
        width,
        height,
    ).view(-1, sample_lattice.shape[0])

    delta = sample_offsets.reshape(-1, 2)
    c = conics.unsqueeze(1).expand(-1, sample_lattice.shape[0], -1).reshape(-1, 3)
    sigmas = (
        0.5 * (c[:, 0] * delta[:, 0] ** 2 + c[:, 2] * delta[:, 1] ** 2)
        + c[:, 1] * delta[:, 0] * delta[:, 1]
    )
    footprint_weights = torch.exp(-sigmas).view(-1, sample_lattice.shape[0])
    return (mask_samples * footprint_weights).sum(dim=1) / footprint_weights.sum(dim=1).clamp_min(1e-6)

class SimpleGaussian:
    def __init__(self, gt_image, gt_depth=None, gt_flow=None, num_points=100000, background="black", sequence_path=None, logs_suffix="_logs", common_logs=True,
                 use_2dgs=False, w_normal_consistency=0.0, w_distortion=0.0,
                 normal_start_frac=0.5, dist_start_frac=0.3,
                 center_scale_world=5e-4, use_meta_uv_debug=False,
                 init_2dgs_first_frame=True,
                 low_memory_2dgs=False,
                 legacy_densify_lr_after_densify=False,
                 force_split_diagnostics=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 2DGS 模式开关及相关参数
        self.use_2dgs = use_2dgs
        self.w_normal_consistency = w_normal_consistency
        self.w_distortion = w_distortion
        self.normal_start_frac = normal_start_frac
        self.dist_start_frac = dist_start_frac
        self.center_scale_world = center_scale_world
        self.use_meta_uv_debug = use_meta_uv_debug
        self.init_2dgs_first_frame = init_2dgs_first_frame
        self.low_memory_2dgs = low_memory_2dgs
        self.legacy_densify_lr_after_densify = legacy_densify_lr_after_densify
        self.force_split_diagnostics = force_split_diagnostics
        if self.use_2dgs:
            print("[2DGS] 模式已启用")
            print(f"[2DGS] w_normal_consistency={w_normal_consistency}, w_distortion={w_distortion}")
            print(f"[2DGS] normal_start_frac={normal_start_frac}, dist_start_frac={dist_start_frac}")
            print(f"[2DGS] init_2dgs_first_frame={init_2dgs_first_frame}")
            print(f"[2DGS] low_memory_2dgs={low_memory_2dgs}")
            print(f"[2DGS] legacy_densify_lr_after_densify={legacy_densify_lr_after_densify}")
            print(f"[2DGS] force_split_diagnostics={force_split_diagnostics}")
        self.gt_image = gt_image.to(self.device)
        self.gt_depth = gt_depth.to(self.device) if gt_depth is not None else None
        self.gt_flow = gt_flow.to(self.device) if gt_flow is not None else None
        self.sequence_path = str(sequence_path) if sequence_path is not None else None
        self.num_points = num_points
        self.clustering = KMeans(n_clusters=2, random_state=0)

        H, W, C = gt_image.shape 
        self.H = H
        self.W = W
        if background == "black":
            self.bg = 0.
        elif background == "white":
            self.bg = 1.
        elif background == "cyan":
            self.bg = 0.33
        else: 
            self.bg =0.
        fov = math.pi / 2.0
        fx = 0.5 * float(W) / math.tan(0.5 * fov)
        fy = 0.5 * float(H) / math.tan(0.5 * fov)
        self.intr = torch.Tensor([fx, fy, float(W) / 2, float(H) / 2]).cuda().float() 
        self.pose = torch.tensor([0., 0., 0., 1., 0., 0., 0.]).cuda().float()
        self.extr = self.get_extr() # world2camera
        # [ ] TODO create a camera class, opt for learning residual of camera pose

        # [ ] TODO support optimize multi-view consistent depth within a window
        
        N = int(num_points)

        def _2dgs(x):
            x = (torch.abs(x) + 1e-8).repeat(1, 2)
            # add the third channel to be zero
            x = torch.cat((x, torch.zeros_like(x[:, :1])), dim=1)
            return x
        
        def _isotropic(x):
            return (torch.abs(x) + 1e-8).repeat(1, 3)

        def sensitive_sigmoid(x, scale=10.):
            return torch.sigmoid(x * scale)
        
        def sensitive_logit(x, scale=10.):
            return torch.logit(x) / scale
        
        self._activations = {
            "scale": lambda x: torch.abs(x),
            "rotate": torch.nn.functional.normalize,
            "opacity": sensitive_sigmoid,
            "rgb": torch.sigmoid
        }
        
        # the inverse of the activation functions
        self._activations_inv = {
            "scale": lambda x: torch.abs(x),
            "rotate": torch.nn.functional.normalize,
            "opacity": sensitive_logit,
            "rgb": torch.logit
        }

        self._attributes = {
            "xyz":      torch.rand((N, 3), dtype=torch.float32).cuda() * 2 - 1,
            # "scale":    torch.rand((N, 1), dtype=torch.float32).cuda(),
            "scale":    torch.rand((N, 3), dtype=torch.float32).cuda(),
            "rotate":   self._activations_inv["rotate"](torch.rand((N, 4), dtype=torch.float32).cuda()), # TODO should be a quaternion
            "opacity":  self._activations_inv["opacity"](0.99*torch.ones((N, 1), dtype=torch.float32).cuda()),
            "rgb":      torch.rand((N, 3), dtype=torch.float32).cuda()
        }

        
        # Get current date and time as string
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Create a directory with the current date and time as its name
        logs_path = "logs"
        if common_logs:
            if logs_suffix is None:
                logs_path = "logs"
            else:
                logs_path = logs_suffix
        else:
            if logs_suffix is None:
                logs_path = str(sequence_path) + "_logs"
            else:
                logs_path = str(sequence_path) + f"_{logs_suffix}"
        log_now_path = os.path.join(logs_path, f"{now}")
        os.makedirs(log_now_path, exist_ok=True)
        log_latest_path = os.path.join(logs_path, "0_latest")
        os.makedirs(log_latest_path, exist_ok=True)
        # make a new directory, always soft link to the current run
        log_now_path_abs = os.path.abspath(log_now_path)
        # 清理 0_latest 目录下的所有内容
        for filename in os.listdir(log_latest_path):
            file_path = os.path.join(log_latest_path, filename)
            try:
                if os.path.islink(file_path) or os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"[Warning] 无法清理路径 {file_path}: {e}")
                
        os.system(f"ln -s {log_now_path_abs} {log_latest_path}/{now}")
        self.dir = log_now_path

    def _render_backend(self, input_group, return_type, center_scale_world=None):
        """统一渲染分发：根据 use_2dgs 切换渲染后端。

        use_2dgs=False 时走原 msplat 路径，True 时走 gsplat 2DGS 路径。
        所有渲染调用都必须走这个 helper，避免 2DGS 分支散落各处。
        """
        if center_scale_world is None:
            center_scale_world = self.center_scale_world
        if self.use_2dgs:
            return render_2dgs.render_multiple_2dgs(
                input_group,
                return_type,
                packed=False,
                sh_degree=0,
                center_scale_world=center_scale_world,
            )
        else:
            return render.render_multiple(input_group, return_type)

    def _make_input_group(self):
        return [
            self.get_attribute("xyz"),
            self.get_attribute("scale"),
            self.get_attribute("rotate"),
            self.get_attribute("opacity"),
            self.get_attribute("rgb"),
            self.intr,
            self.get_extr(),
            self.bg,
            self.W,
            self.H,
        ]

    # [ ] TODO @property
    def get_extr(self):
        # normalize rotation
        Q = self.pose[:4]
        T = utils.signed_expm1(self.pose[4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous() # (4, 4)
        extr = RT[:3, :] # (3, 4)
        return extr

    def add_optimizer(self, lr=1e-2, lr_camera=0., exclude_key=None, camera_only=False, depth_invariant=True):
        self.lr = lr
        self.lr_camera = lr_camera
        for attribute_name in self._attributes.keys():
            if attribute_name != exclude_key:
                self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        # self.extr = nn.Parameter(self.extr).requires_grad_(True)
        self.pose = nn.Parameter(self.pose).requires_grad_(True)

        # assign different lr to attributes and extr
        optim_params=[
            {"params": self._attributes.values(), "lr": lr, "name": "attributes"},
            # {"params": self.extr, "lr": lr_camera, "name": "extr"},
            {"params": self.pose, "lr": lr_camera, "name": "extr"},
        ]

        if hasattr(self, 'pose_list'):
            for idx in range(len(self.pose_list)):
                # self.pose_list[idx] = nn.Parameter(self.pose_list[idx]).requires_grad_(True)
                optim_params.append({"params": self.pose_list[idx], "lr": lr_camera, "name": f"extr_{idx}"})

        if depth_invariant:
            self.depth_a = nn.Parameter(torch.ones(1).to(self.device)).requires_grad_(True)
            self.depth_b = nn.Parameter(torch.zeros(1).to(self.device)).requires_grad_(True)
            optim_params.append({"params": self.depth_a, "lr": lr, "name": "depth_a"})
            optim_params.append({"params": self.depth_b, "lr": lr, "name": "depth_b"})
        else:
            self.depth_a = nn.Parameter(torch.ones(1).to(self.device)).requires_grad_(False)
            self.depth_b = nn.Parameter(torch.zeros(1).to(self.device)).requires_grad_(False)
        
        self.optimizer = torch.optim.Adam(optim_params)

    def _append_optimizer_tensor(self, attribute_name, extension):
        old_param = self._attributes[attribute_name]
        new_param = nn.Parameter(
            torch.cat((old_param.detach(), extension.detach()), dim=0)
        ).requires_grad_(True)

        if not hasattr(self, "optimizer"):
            self._attributes[attribute_name] = new_param
            return

        for group in self.optimizer.param_groups:
            for param_idx, param in enumerate(group["params"]):
                if param is not old_param:
                    continue

                stored_state = self.optimizer.state.pop(param, None)
                group["params"][param_idx] = new_param
                if stored_state is not None:
                    for key, value in list(stored_state.items()):
                        if torch.is_tensor(value) and value.shape == old_param.shape:
                            state_extension = torch.zeros(
                                (extension.shape[0], *value.shape[1:]),
                                dtype=value.dtype,
                                device=value.device,
                            )
                            stored_state[key] = torch.cat((value, state_extension), dim=0)
                    self.optimizer.state[new_param] = stored_state
                self._attributes[attribute_name] = new_param
                return

        self._attributes[attribute_name] = new_param

    def set_gt_image(self, gt_image):
        self.gt_image = gt_image.to(self.device)

    def set_gt_depth(self, gt_depth):
        self.gt_depth = gt_depth.to(self.device)

    def set_gt_flow(self, gt_flow):
        self.gt_flow = gt_flow.to(self.device)

    def _merge_opacity_hard_cap(self, opacity_hard_cap, indices, cap):
        if indices.numel() == 0:
            return opacity_hard_cap
        if opacity_hard_cap is None:
            opacity_hard_cap = torch.ones(
                self.current_pts_num(), dtype=torch.float32, device=self.device
            )
        opacity_hard_cap[indices] = torch.minimum(
            opacity_hard_cap[indices],
            torch.full_like(opacity_hard_cap[indices], cap),
        )
        return opacity_hard_cap

    def _apply_opacity_hard_cap(self, opacity_hard_cap):
        if opacity_hard_cap is None:
            return
        hard_mask = opacity_hard_cap < 1.0
        if not hard_mask.any():
            return
        with torch.no_grad():
            opacity = self.get_attribute("opacity")
            capped_opacity = torch.minimum(opacity, opacity_hard_cap.unsqueeze(-1))
            capped_opacity = torch.clamp(capped_opacity, min=1e-4, max=1.0 - 1e-4)
            self._attributes["opacity"].data.copy_(
                self._activations_inv["opacity"](capped_opacity)
            )

            state = self.optimizer.state.get(self._attributes["opacity"], None)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    value = state.get(key, None)
                    if torch.is_tensor(value) and value.shape[:1] == hard_mask.shape[:1]:
                        value[hard_mask] = 0

    def load_camera(self, focal=None, pp=None, extr=None, scale=None, show=True):
        if focal is not None:
            focal_tensor = torch.as_tensor(focal, dtype=torch.float32, device=self.device).flatten()
            if focal_tensor.numel() == 1:
                focal_tensor = focal_tensor.repeat(2)
            self.intr[:2] = focal_tensor[:2]
        if pp is not None:
            self.intr[2:] = torch.tensor(pp, dtype=torch.float32).to(self.device)
        if extr is not None:
            R = torch.tensor(extr[:3, :3]).to(self.device)
            T = torch.tensor(extr[:3, 3]).to(self.device)
            if scale is not None:
                # print("[camera] T before: ", T)
                T = T * scale
                # print("[camera] T after: ", T)
            pose = self.pose.detach().clone()
            pose[0:4] = roma.rotmat_to_unitquat(R)
            pose[4:7] = utils.signed_log1p(T)
            self.pose = nn.Parameter(pose).requires_grad_(True)
        if show:
            print("[camera] intr: ", self.intr)
            # print("[camera] extr: \n", self.extr)
            print("[camera] extr: \n", self.get_extr())
    
    def choose_idx(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, 3)
        return idx

    def choose_extr(self, idx=None):
        if idx is None:
            # random choose one from self.pose_list
            idx = np.random.randint(0, len(self.pose_list))
        Q = self.pose_list[idx][:4]
        T = utils.signed_expm1(self.pose_list[idx][4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous() # (4, 4)
        extr = RT[:3, :] # (3, 4)

        return extr
    
    def choose_move_mask(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self.move_masks))
        return self.move_masks[idx]
    
    def init_gaussians_from_image(self, gt_image, gt_depth=None, num_points=None, mask=None, drop_to=None):
        # gt_image (H, W, C) in [0, 1]
        if num_points is None:
            num_points = self.num_points
        xys, depths, scales, rgbs, gt_depth = utils.complex_texture_sampling(gt_image, gt_depth, num_points=num_points, device=self.device, mask=mask, drop_to=drop_to)
        new_num_points = xys.shape[0]
        """depths: 0 is near, 1 is far"""
        xys = torch.from_numpy(xys).to(self.device).float()
        depths = depths.to(self.device).float()
        self.gt_depth = gt_depth.to(self.device).float()

        self._attributes["xyz"] = geometry.pix2world(xys, depths, self.intr, self.get_extr())

        print("[init] x range: ", self._attributes["xyz"][:,0].min().item(), self._attributes["xyz"][:,0].max().item())
        print("[init] y range: ", self._attributes["xyz"][:,1].min().item(), self._attributes["xyz"][:,1].max().item())
        print("[init] z range: ", self._attributes["xyz"][:,2].min().item(), self._attributes["xyz"][:,2].max().item())

        scales = scales * (depths/depths.min()).squeeze().cpu().numpy()
        scales = torch.from_numpy(scales).float().unsqueeze(1).repeat(1, 3).to(self.device)
        scales = torch.clamp(scales, max=1e-3)
        if self.use_2dgs and self.init_2dgs_first_frame:
            scales = self._make_2dgs_scales(scales)
        self._attributes["scale"] = self._activations_inv["scale"](scales)

        rgbs = torch.from_numpy(rgbs).float().contiguous().to(self.device)
        eps = 1e-6  # float32-safe clamp for torch.logit
        rgbs = torch.clamp(rgbs, min=eps, max=1-eps)
        # calculate the inverse of sigmoid function, i.e., logit function
        self._attributes["rgb"] = self._activations_inv["rgb"](rgbs)

        opacity = 0.99 * torch.ones((new_num_points, 1), device=self.device).float()
        self._attributes["opacity"] = self._activations_inv["opacity"](opacity)

        if self.use_2dgs and self.init_2dgs_first_frame:
            normal_map = self._build_depth_normal_map(self.gt_depth)
            sample_x = xys[:, 0].long().clamp(0, self.W - 1)
            sample_y = xys[:, 1].long().clamp(0, self.H - 1)
            rotate = self._quats_from_normals(normal_map[sample_y, sample_x])
        else:
            rotate = torch.rand((new_num_points, 4), dtype=torch.float32).cuda()
        self._attributes["rotate"] = self._activations_inv["rotate"](rotate)
        
    def current_pts_num(self):
        return self._attributes["xyz"].shape[0]
    
    def get_attribute(self, name):
        try:
            if name in self._activations.keys() and self._activations[name] is not None:
                return self._activations[name](self._attributes[name])
            else:
                return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")

    def save_checkpoint(self, ckpt_name=None, camera_only=False):
        # save checkpoint
        checkpoint = {
            "attributes": self._attributes,
            "intr": self.intr,
            # "extr": self.extr,
            "extr": self.get_extr(),
            "still_mask": self.still_mask if hasattr(self, 'still_mask') else None,
            "move_seg": self.move_seg if hasattr(self, 'move_seg') else None,
            "last_uv": self.last_uv if hasattr(self, 'last_uv') else None,
            "width": self.W,
            "height": self.H,
        }
        if hasattr(self, 'pose_list'):
            checkpoint["pose_list"] = self.pose_list
        if ckpt_name is None:
            ckpt_name = 'ckpt'
        # make directory
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)
        self.checkpoint_path = os.path.join(self.dir, "ckpt", f"{ckpt_name}.tar")
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path, show=True):
        checkpoint = torch.load(checkpoint_path)
        self._attributes = checkpoint["attributes"]
        self.intr = checkpoint["intr"]
        self.extr = checkpoint["extr"]
        # self.pose = self.load_camera(extr=self.extr)
        self.load_camera(extr=self.extr, show=show)
        if "still_mask" in checkpoint.keys():
            self.still_mask = checkpoint["still_mask"]
        if "move_seg" in checkpoint.keys():
            self.move_seg = checkpoint["move_seg"]
        if "last_uv" in checkpoint.keys():
            self.last_uv = checkpoint["last_uv"]
        del checkpoint
        torch.cuda.empty_cache()

    def init_mask_prompt_pts(self, mask_prompt, ckpt_name):
        input_group = [
            self.get_attribute("xyz"),
            self.get_attribute("scale"),
            self.get_attribute("rotate"),
            self.get_attribute("opacity"),
            self.get_attribute("rgb"),
            self.intr,
            # self.extr,
            self.get_extr(),
            self.bg,
            self.W,
            self.H,
        ]

        return_dict = self._render_backend(
            input_group,
            ["uv", "center"]
        )
        uv = return_dict["uv"].detach() # (N, 2)
        uv_within = (uv[:,0] > 0) & (uv[:,0] < self.W-1) & (uv[:,1] > 0) & (uv[:,1] < self.H-1) # (N,)
        uv = uv[uv_within] # (N_within, 2)
        # print(uv[:,1][:10])
        y_coords = uv[:,1].long() # (N_within,)
        x_coords = uv[:,0].long() # (N_within,)
        # print(y_coords[:10])
        print(y_coords.max(), y_coords.min())
        print(x_coords.max(), x_coords.min())
        print(mask_prompt.shape)

        mask_np = mask_prompt.detach().cpu().numpy() * 255
        mask_np = mask_np.astype(np.uint8)
        # save
        os.makedirs(os.path.join(self.dir, "images_seg"), exist_ok=True)
        imageio.imwrite(os.path.join(self.dir, "images_seg", f"propagate_mask_{ckpt_name}.png"), mask_np)


        # select the points that are within the mask
        mask_prompt_pts = mask_prompt[y_coords, x_coords] # (N_within,)
        self.mask_prompt_pts = uv_within.clone()
        self.mask_prompt_pts[uv_within] = mask_prompt_pts

    def save_geometry_eval(self, frame_name, return_dict):
        """Persist raw expected depth and opacity for one optimized frame.

        This is an opt-in, post-render export.  It intentionally receives the
        already-computed no-grad render dictionary so it cannot affect losses,
        gradients, optimizer state, or densification.
        """
        depth_expected = return_dict.get("depth_expected")
        alpha = return_dict.get("alpha")
        if depth_expected is None or alpha is None:
            raise RuntimeError(
                "Geometry export requires renderer outputs 'depth_expected' and 'alpha'"
            )

        def _to_hw(value, label):
            if not torch.is_tensor(value):
                raise TypeError(f"Renderer output {label!r} is not a tensor")
            array = value.detach().float().cpu().numpy()
            array = np.squeeze(array)
            if array.ndim != 2:
                raise ValueError(f"Renderer output {label!r} has shape {array.shape}, expected (H,W)")
            return array.astype(np.float32, copy=False)

        depth_array = _to_hw(depth_expected, "depth_expected")
        alpha_array = _to_hw(alpha, "alpha")
        if depth_array.shape != alpha_array.shape:
            raise ValueError(
                f"Geometry output shape mismatch: depth={depth_array.shape}, alpha={alpha_array.shape}"
            )

        raw_dir = Path(self.dir) / "geometry_raw"
        depth_dir = raw_dir / "depth_expected"
        alpha_dir = raw_dir / "alpha"
        depth_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir.mkdir(parents=True, exist_ok=True)
        stem = str(frame_name)
        if Path(stem).name != stem:
            raise ValueError(f"Invalid geometry frame name: {frame_name!r}")
        for suffix in (".png", ".jpg", ".jpeg", ".npy"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if not stem or stem in {".", ".."}:
            raise ValueError(f"Invalid geometry frame name: {frame_name!r}")
        depth_rel = Path("depth_expected") / f"{stem}.npy"
        alpha_rel = Path("alpha") / f"{stem}.npy"
        np.save(raw_dir / depth_rel, depth_array)
        np.save(raw_dir / alpha_rel, alpha_array)

        manifest_path = raw_dir / "geometry_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                manifest = {}
        else:
            manifest = {}
        if not isinstance(manifest, dict) or manifest.get("format") != "gflow-geometry-v1":
            manifest = {
                "format": "gflow-geometry-v1",
                "version": 1,
                "backend": "2dgs" if self.use_2dgs else "msplat_3dgs",
                "depth_semantics": "expected_camera_z",
                "alpha_semantics": "accumulated_opacity",
                "sequence_path": self.sequence_path,
                "width": int(depth_array.shape[1]),
                "height": int(depth_array.shape[0]),
                "frames": [],
            }
        frames = manifest.setdefault("frames", [])
        if not isinstance(frames, list):
            frames = []
            manifest["frames"] = frames
        existing = next(
            (item for item in frames if isinstance(item, dict) and str(item.get("frame_name", "")) == stem),
            None,
        )
        if existing is not None:
            try:
                frame_index = int(existing["frame_index"])
            except (KeyError, TypeError, ValueError):
                frame_index = len(frames)
        else:
            try:
                frame_index = int(stem)
            except ValueError:
                frame_index = len(frames)
        entry = {
            "frame_index": frame_index,
            "frame_name": stem,
            "depth_expected": depth_rel.as_posix(),
            "alpha": alpha_rel.as_posix(),
            "height": int(depth_array.shape[0]),
            "width": int(depth_array.shape[1]),
            "render_intrinsics": [float(value) for value in self.intr.detach().cpu().flatten().tolist()],
        }
        frames[:] = [item for item in frames if not (isinstance(item, dict) and str(item.get("frame_name", "")) == stem)]
        frames.append(entry)
        frames.sort(key=lambda item: (int(item.get("frame_index", 10**18)) if isinstance(item, dict) else 10**18, str(item)))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def train(self, iterations=500, lr=1e-2, lr_camera=0., lambda_rgb=1.,
              lambda_depth=0., lambda_flow=0., lambda_refresh=0., lambda_stale_opacity=0.,
              lambda_var=0., lambda_still=0., lambda_scale=0.,
              save_imgs=False, save_videos=False, save_ckpt=False, move_mask=None,
              save_geometry_eval=False,
              ckpt_name="ckpt", densify_interval=500, densify_times=1, densify_iter=0,
              grad_threshold=5e-3, mask=None, camera_only=False, eps=10, min_samples=20,
              densify_occ_percent=0.1, densify_err_thre=1e-2, densify_err_percent=0.2,
              global_step_offset=0, global_total_steps=None, global_start_time=None,
              stale_opacity_hard_cap=1.0, orphan_move_opacity_hard_cap=1.0):
        frames = []
        frames_depth = []
        frames_center = []
        progress_bar = tqdm(range(iterations), desc="Training")
        train_start_time = time.time()
        l1_loss = nn.SmoothL1Loss(reduction="none")
        mse_loss = nn.MSELoss()
        mse_loss_pixel = nn.MSELoss(reduction='none')
        utils.SSIM_loss = utils.SSIM()
        refresh_mask = None
        refresh_move_mask = None
        refresh_region_mask = None
        stale_mask = None
        relabel_move_mask = None
        refresh_gt_image = None
        refresh_warmup_iters = max(1, int(iterations * 0.2))
        stale_warmup_iters = max(1, int(iterations * 0.2))
        refresh_still_gate_threshold = 0.25
        refresh_still_loss_weight = 0.35
        stale_mask_tensor = None
        refresh_move_mask_tensor = None
        refresh_region_mask_tensor = None
        relabel_move_mask_tensor = None
        stale_opacity_hard_cap = float(max(0.0, min(1.0, stale_opacity_hard_cap)))
        orphan_move_opacity_hard_cap = float(max(0.0, min(1.0, orphan_move_opacity_hard_cap)))
        if move_mask is not None:
            if move_mask.dim() == 3:
                move_mask = move_mask.sum(dim=-1)
            move_mask = (move_mask.squeeze() > 0).to(self.device)
        if mask is not None:
            occ_mask = mask
            if occ_mask.dim() == 3:
                occ_mask = occ_mask.sum(dim=-1)
            occ_mask = (occ_mask.squeeze() > 0).to(self.device)
            # Use a slightly eroded disocclusion mask so refresh/stale suppression only act on cleaner interior regions.
            refresh_mask = (
                torch.nn.functional.avg_pool2d(
                    occ_mask.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ) > (1.0 - 1e-6)
            ).squeeze(0).squeeze(0)
            # Stale suppression needs a wider catchment area than refresh because stale splats
            # often linger around disocclusion boundaries instead of the clean interior.
            stale_mask = (
                torch.nn.functional.max_pool2d(
                    occ_mask.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ) > 0
            ).squeeze(0).squeeze(0)
            stale_mask_tensor = stale_mask.float().unsqueeze(0).unsqueeze(0)
        if move_mask is not None:
            refresh_move_mask = (
                torch.nn.functional.max_pool2d(
                    move_mask.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=15,
                    stride=1,
                    padding=7,
                ) > 0
            ).squeeze(0).squeeze(0)
            refresh_move_mask_tensor = refresh_move_mask.float().unsqueeze(0).unsqueeze(0)
            relabel_move_mask = refresh_move_mask.clone()
            if stale_mask is not None:
                relabel_move_mask = relabel_move_mask | stale_mask
            relabel_move_mask_tensor = relabel_move_mask.float().unsqueeze(0).unsqueeze(0)
        if refresh_mask is not None:
            refresh_region_mask = refresh_mask.clone()
        if refresh_move_mask is not None:
            refresh_region_mask = (
                refresh_move_mask.clone()
                if refresh_region_mask is None
                else (refresh_region_mask | refresh_move_mask)
            )
        if refresh_region_mask is not None:
            refresh_region_mask_tensor = refresh_region_mask.float().unsqueeze(0).unsqueeze(0)
        if self.use_2dgs and not camera_only and lambda_refresh > 0:
            refresh_gt_image = self.gt_image.permute(2, 0, 1).unsqueeze(0)

        """pre-update prosessing"""
        if not camera_only and hasattr(self, 'still_mask'):
            uv_move = self.last_uv[:self.last_still_mask.shape[0]][~self.last_still_mask] # (N_move, 2)
            # within the image
            uv_within_index = (uv_move[:,0] > 0) & (uv_move[:,0] < self.W-1) & (uv_move[:,1] > 0) & (uv_move[:,1] < self.H-1) # (N_move,)
            uv_move = uv_move[uv_within_index] # (N_move_within, 2)
            y_coords = uv_move[:,1].long().clamp(0, self.gt_flow.shape[0] - 1) # (N_move_within,)
            x_coords = uv_move[:,0].long().clamp(0, self.gt_flow.shape[1] - 1) # (N_move_within,)
            move_flow = self.gt_flow[y_coords, x_coords] # (N_move_within, 2)
            uv_move = uv_move + move_flow  # (N_move_within, 2)
            y_coords_move = uv_move[:,1].long() # (N_move_within,)
            x_coords_move = uv_move[:,0].long() # (N_move_within,)
            # clamp the coords using the actual shape of gt_depth instead of initial H/W
            y_coords_move = torch.clamp(y_coords_move, 0, self.gt_depth.shape[0]-1)
            x_coords_move = torch.clamp(x_coords_move, 0, self.gt_depth.shape[1]-1)
            depth_move = self.gt_depth[y_coords_move, x_coords_move] # (N_move_within,)
            # xyz_move = geometry.pix2world(uv_move, depth_move, self.intr, self.extr) # (N_move_within, 3)
            xyz_move = geometry.pix2world(uv_move, depth_move, self.intr, self.get_extr()) # (N_move_within, 3)
            # assign the new xyz to the moving part
            xyz = self._attributes["xyz"].clone()

            temp_1 = xyz[:self.last_still_mask.shape[0]][~self.last_still_mask].clone()
            temp_1[uv_within_index] = xyz_move.detach()

            temp_2 = xyz[:self.last_still_mask.shape[0]].clone()
            temp_2[~self.last_still_mask] = temp_1

            xyz[:self.last_still_mask.shape[0]] = temp_2

            self._attributes["xyz"] = xyz
            # xyz = self._attributes["xyz"].clone()
            # temp_xyz = xyz[:self.last_still_mask.shape[0]].detach()
            # temp_xyz[~self.last_still_mask] = xyz_move.detach()
            # xyz[:self.last_still_mask.shape[0]] = temp_xyz
            # self._attributes["xyz"] = xyz

        self.add_optimizer(lr, lr_camera, exclude_key=None, camera_only=camera_only, depth_invariant=True)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=iterations)
        self._freeze_scheduler_after_densify = False

        """iterate the optimization"""
        for iteration in range(0, iterations):
            loss = 0.
            opacity_hard_cap = None

            input_group = self._make_input_group()

            need_surf_normal_cache = (
                self.use_2dgs
                and not camera_only
                and (
                    (iteration == 0 and hasattr(self, 'last_xyz'))
                    or (
                        densify_interval
                        and (iteration + 1) % densify_interval == 0
                        and (iteration + 1) // densify_interval <= densify_times
                    )
                )
            )
            _return_type = train_render_return_types(
                use_2dgs=self.use_2dgs,
                camera_only=camera_only,
                iteration=iteration,
                iterations=iterations,
                w_normal_consistency=self.w_normal_consistency,
                w_distortion=self.w_distortion,
                normal_start_frac=self.normal_start_frac,
                dist_start_frac=self.dist_start_frac,
                need_surf_normal_cache=need_surf_normal_cache,
                low_memory_2dgs=self.low_memory_2dgs,
            )
            return_dict = self._render_backend(
                input_group,
                _return_type
            )

            # render image
            rendered_rgb, uv, depth = return_dict["rgb"], return_dict["uv"], return_dict["depth"]

            # render depth map
            rendered_depth_map = return_dict.get("depth_map")

            rendered_depth_map_color = return_dict.get("depth_map_color")
            rendered_center = return_dict.get("center")

            progress_dict = {}
            if self.use_2dgs and not camera_only and lambda_refresh > 0:
                progress_dict["refresh"] = "0.000000"
                progress_dict["refresh_pts"] = "0"
            if self.use_2dgs and not camera_only and lambda_stale_opacity > 0:
                progress_dict["stale_opacity"] = "0.000000"
                progress_dict["stale_pts"] = "0"

            # valid_uv = uv within the a HxW range
            valid_uv_index = ((uv[:,0] > 0) & (uv[:,0] < self.W-1) & (uv[:,1] > 0) & (uv[:,1] < self.H-1)) # (N,)
            self.within_index = valid_uv_index
            # [ ] TODO do not calculate loss on the moving part
            if hasattr(self, 'still_mask_tentative') and camera_only and not self.use_2dgs: # if camera_only, combine the input move mask and mask of moving gs.
                # """
                input_group_temp = [
                    self.get_attribute("xyz").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.get_attribute("scale").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.get_attribute("rotate").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.get_attribute("opacity").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.get_attribute("rgb").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.intr,
                    self.get_extr(),
                    # self.extr,
                    self.bg,
                    self.W,
                    self.H,
                ]

                return_dict_temp = self._render_backend(
                    input_group_temp,
                    ["rgb"]
                )
                move_gs_rgb = return_dict_temp["rgb"] # shape (3, H, W)
                # print(move_gs_rgb.max(), move_gs_rgb.min())
                move_gs_grey = 0.299 * move_gs_rgb[0, :, :] + 0.587 * move_gs_rgb[1, :, :] + 0.114 * move_gs_rgb[2, :, :] # shape (H, W)
                move_gs_mask = move_gs_grey > 0.
                move_mask = move_gs_mask | move_mask
            if lambda_rgb > 0:
                if camera_only:
                    rendered_rgb = rendered_rgb * ~move_mask.unsqueeze(0)
                    gt_image = self.gt_image * ~move_mask.unsqueeze(-1)
                else:
                    gt_image = self.gt_image

                loss_rgb_pixel = mse_loss_pixel(rendered_rgb.permute(1, 2, 0), gt_image).mean(dim=2)
                loss_rgb = loss_rgb_pixel.mean()
                loss_ssim = 1-utils.SSIM_loss(rendered_rgb.unsqueeze(0), gt_image.permute(2, 0, 1).unsqueeze(0))
                loss_rgb = loss_rgb + loss_ssim
                loss += lambda_rgb * loss_rgb
                progress_dict["rgb"] = f"{loss_rgb.item():.6f}"

            if rendered_depth_map is not None:
                rendered_depth_map = rendered_depth_map.permute(1, 2, 0)
            if hasattr(self, 'still_mask'):
                if camera_only: # only optimize the still part
                    valid_uv_index[:self.still_mask.shape[0]] = self.still_mask & valid_uv_index[:self.still_mask.shape[0]]
                else: # only optimize the move part
                    valid_uv_index[:self.still_mask.shape[0]] = ~self.still_mask & valid_uv_index[:self.still_mask.shape[0]]
            valid_uv = uv[valid_uv_index]
            # 增加防御性边界卡扣(Clamp)，使用正被读取目标张量自身的尺寸判定，杜绝数据集中异常缩放导致越界
            y_idx = valid_uv[:,1].long().clamp(0, self.gt_depth.shape[0] - 1)
            x_idx = valid_uv[:,0].long().clamp(0, self.gt_depth.shape[1] - 1)
            depth_point_gt = self.gt_depth[y_idx, x_idx] # shape (N, 1)
            depth_point = depth[valid_uv_index] # shape (N, 1)

            if lambda_depth > 0:

                if rendered_depth_map is not None:
                    # rendered_depth_map_norm = rendered_depth_map
                    rendered_depth_map_norm = self.depth_a * rendered_depth_map + self.depth_b # scale and shift invariant depth loss
                    gt_depth_norm = self.gt_depth

                    loss_depth = mse_loss_pixel(rendered_depth_map_norm, gt_depth_norm) / (rendered_depth_map_norm + gt_depth_norm)
                    if camera_only:
                        loss_depth = loss_depth * ~move_mask.unsqueeze(-1)
                    loss_depth = loss_depth.mean()
                else:
                    if depth_point.numel() == 0:
                        loss_depth = torch.zeros((), dtype=rendered_rgb.dtype, device=self.device)
                    else:
                        depth_point_norm = self.depth_a * depth_point + self.depth_b
                        loss_depth = mse_loss_pixel(depth_point_norm, depth_point_gt) / (depth_point_norm + depth_point_gt).clamp_min(1e-5)
                        loss_depth = loss_depth.mean()

                loss += lambda_depth * loss_depth
                progress_dict["depth"] = f"{loss_depth.item():.6f}"

            if lambda_var: # penalize the scales with large variance, to avoid needle-like artifacts
                loss_var = torch.mean(torch.std(self.get_attribute("scale"), dim=1))
                loss += lambda_var * loss_var
                progress_dict["var"] = f"{loss_var.item():.6f}"

            if lambda_scale: # penalize the gaussian points with large scales, by L2 loss
                loss_scale = torch.norm(self.get_attribute("scale")[self.within_index], dim=1)
                depth_norm = 1 / depth_point
                # import pdb; pdb.set_trace()
                loss_scale = loss_scale * depth_norm.squeeze()
                loss_scale = loss_scale.mean()
                loss += lambda_scale * loss_scale
                progress_dict["scale"] = f"{loss_scale.item():.6f}"


            if lambda_still and hasattr(self, 'still_mask'): # pushing still gaussians to be still
                still_shape = self.last_still_mask.shape[0]
                loss_still = torch.norm(self.get_attribute("xyz")[:still_shape][self.last_still_mask] - self.last_xyz[:still_shape][self.last_still_mask], dim=1).mean()
                loss += lambda_still * loss_still
                progress_dict["still"] = f"{loss_still.item():.6f}"

            if (self.use_2dgs and not camera_only and lambda_refresh > 0
                    and refresh_region_mask_tensor is not None
                    and hasattr(self, "last_uv")
                    and hasattr(self, "last_num")
                    and hasattr(self, "last_still_mask")):
                old_point_num = min(self.last_num, self.last_still_mask.shape[0], uv.shape[0], input_group[4].shape[0])
                if old_point_num > 0:
                    refresh_uv = uv[:old_point_num]
                    refresh_rgb = input_group[4][:old_point_num]
                    old_moving_mask = ~self.last_still_mask[:old_point_num]
                    meta_radii = None
                    meta_conics = None
                    meta_means2d = None
                    meta = return_dict.get("meta", None)
                    if meta is not None:
                        meta_radii = meta.get("radii", None)
                        meta_conics = meta.get("conics", None)
                        meta_means2d = meta.get("means2d", None)
                        if meta_radii is not None and meta_radii.dim() == 3 and meta_radii.shape[0] == 1:
                            meta_radii = meta_radii.squeeze(0)
                        if meta_conics is not None and meta_conics.dim() == 3 and meta_conics.shape[0] == 1:
                            meta_conics = meta_conics.squeeze(0)
                        if meta_means2d is not None and meta_means2d.dim() == 3 and meta_means2d.shape[0] == 1:
                            meta_means2d = meta_means2d.squeeze(0)
                        if meta_radii is not None and meta_radii.shape[0] < old_point_num:
                            meta_radii = None
                        if meta_conics is not None and meta_conics.shape[0] < old_point_num:
                            meta_conics = None
                        if meta_means2d is not None and meta_means2d.shape[0] < old_point_num:
                            meta_means2d = None
                    refresh_center_uv = refresh_uv
                    if meta_means2d is not None:
                        refresh_center_uv = meta_means2d[:old_point_num].to(refresh_uv.dtype)
                    refresh_valid = (
                        (refresh_center_uv[:, 0] >= 0) & (refresh_center_uv[:, 0] <= self.W - 1) &
                        (refresh_center_uv[:, 1] >= 0) & (refresh_center_uv[:, 1] <= self.H - 1)
                    )
                    if refresh_valid.any():
                        refresh_uv = refresh_uv[refresh_valid]
                        refresh_center_uv = refresh_center_uv[refresh_valid]
                        refresh_rgb = refresh_rgb[refresh_valid]
                        old_moving_mask = old_moving_mask[refresh_valid]
                        move_contour_hit = torch.zeros_like(old_moving_mask)
                        if refresh_move_mask_tensor is not None:
                            move_contour_hit = _sample_mask_weights(
                                refresh_move_mask_tensor,
                                refresh_center_uv,
                                self.W,
                                self.H,
                            ) > 0.05
                        gated_still_mask = torch.zeros_like(old_moving_mask)
                        if stale_mask_tensor is not None and move_contour_hit.any():
                            old_still_mask = ~old_moving_mask
                            stale_refresh_mask = old_still_mask & move_contour_hit
                            if stale_refresh_mask.any():
                                stale_centers = refresh_center_uv[stale_refresh_mask]
                                if meta_radii is not None and meta_conics is not None:
                                    stale_radii = meta_radii[:old_point_num][refresh_valid][stale_refresh_mask].to(refresh_center_uv.dtype)
                                    stale_conics = meta_conics[:old_point_num][refresh_valid][stale_refresh_mask].to(refresh_center_uv.dtype)
                                    stale_refresh_ratio = _compute_footprint_overlap_ratio(
                                        stale_mask_tensor,
                                        stale_centers,
                                        stale_radii,
                                        stale_conics,
                                        self.W,
                                        self.H,
                                    )
                                else:
                                    stale_refresh_ratio = _sample_mask_weights(
                                        stale_mask_tensor,
                                        stale_centers,
                                        self.W,
                                        self.H,
                                    )
                                gated_still_mask[stale_refresh_mask] = stale_refresh_ratio > refresh_still_gate_threshold
                        refresh_candidate_mask = old_moving_mask | gated_still_mask
                        if refresh_candidate_mask.any():
                            refresh_uv = refresh_uv[refresh_candidate_mask]
                            refresh_center_uv = refresh_center_uv[refresh_candidate_mask]
                            refresh_rgb = refresh_rgb[refresh_candidate_mask]
                            refresh_loss_weights = torch.where(
                                old_moving_mask[refresh_candidate_mask],
                                torch.ones_like(refresh_rgb[:, 0]),
                                torch.full_like(refresh_rgb[:, 0], refresh_still_loss_weight),
                            )
                            refresh_mask_hit = _sample_mask_weights(
                                refresh_region_mask_tensor,
                                refresh_center_uv,
                                self.W,
                                self.H,
                            ) > 0.05
                            if refresh_mask_hit.any():
                                refresh_uv = refresh_uv[refresh_mask_hit]
                                refresh_rgb = refresh_rgb[refresh_mask_hit]
                                refresh_loss_weights = refresh_loss_weights[refresh_mask_hit]
                                refresh_grid = refresh_uv.clone()
                                refresh_grid[:, 0] = 2.0 * (refresh_grid[:, 0] / max(self.W - 1, 1)) - 1.0
                                refresh_grid[:, 1] = 2.0 * (refresh_grid[:, 1] / max(self.H - 1, 1)) - 1.0
                                refresh_grid = refresh_grid.clamp(-1.0, 1.0).view(1, -1, 1, 2)
                                refresh_target = torch.nn.functional.grid_sample(
                                    refresh_gt_image,
                                    refresh_grid,
                                    mode="bilinear",
                                    padding_mode="border",
                                    align_corners=True,
                                ).squeeze(0).squeeze(-1).permute(1, 0)
                                loss_refresh_point = torch.nn.functional.smooth_l1_loss(
                                    refresh_rgb,
                                    refresh_target,
                                    beta=0.1,
                                    reduction="none",
                                )
                                loss_refresh = (
                                    loss_refresh_point.mean(dim=1) * refresh_loss_weights
                                ).sum() / refresh_loss_weights.sum().clamp_min(1e-6)
                                if refresh_warmup_iters <= 1:
                                    refresh_warmup = 1.0
                                else:
                                    refresh_warmup = min(1.0, iteration / float(refresh_warmup_iters - 1))
                                loss += lambda_refresh * refresh_warmup * loss_refresh
                                progress_dict["refresh"] = f"{loss_refresh.item():.6f}"
                                progress_dict["refresh_pts"] = str(int(refresh_mask_hit.sum().item()))

            if (self.use_2dgs and not camera_only and lambda_stale_opacity > 0
                    and stale_mask_tensor is not None
                    and hasattr(self, "last_uv")
                    and hasattr(self, "last_num")):
                old_point_num = min(self.last_num, uv.shape[0], input_group[3].shape[0])
                if old_point_num > 0:
                    stale_uv = uv[:old_point_num]
                    stale_opacity = input_group[3][:old_point_num].squeeze(-1)
                    meta_radii = None
                    meta_conics = None
                    meta_means2d = None
                    meta = return_dict.get("meta", None)
                    if meta is not None:
                        meta_radii = meta.get("radii", None)
                        meta_conics = meta.get("conics", None)
                        meta_means2d = meta.get("means2d", None)
                        if meta_radii is not None and meta_radii.dim() == 3 and meta_radii.shape[0] == 1:
                            meta_radii = meta_radii.squeeze(0)
                        if meta_conics is not None and meta_conics.dim() == 3 and meta_conics.shape[0] == 1:
                            meta_conics = meta_conics.squeeze(0)
                        if meta_means2d is not None and meta_means2d.dim() == 3 and meta_means2d.shape[0] == 1:
                            meta_means2d = meta_means2d.squeeze(0)
                        if meta_radii is not None and meta_radii.shape[0] < old_point_num:
                            meta_radii = None
                        if meta_conics is not None and meta_conics.shape[0] < old_point_num:
                            meta_conics = None
                        if meta_means2d is not None and meta_means2d.shape[0] < old_point_num:
                            meta_means2d = None
                    stale_center_uv = stale_uv
                    if meta_means2d is not None:
                        stale_center_uv = meta_means2d[:old_point_num].to(stale_uv.dtype)
                    stale_valid = (
                        (stale_center_uv[:, 0] >= 0) & (stale_center_uv[:, 0] <= self.W - 1) &
                        (stale_center_uv[:, 1] >= 0) & (stale_center_uv[:, 1] <= self.H - 1)
                    )
                    if meta_radii is not None:
                        stale_valid = stale_valid & (meta_radii[:old_point_num] > 0).all(dim=-1)
                    if stale_valid.any():
                        stale_center_uv = stale_center_uv[stale_valid]
                        stale_opacity = stale_opacity[stale_valid]
                        if meta_radii is not None and meta_conics is not None:
                            stale_radii = meta_radii[:old_point_num][stale_valid].to(stale_center_uv.dtype)
                            stale_conics = meta_conics[:old_point_num][stale_valid].to(stale_center_uv.dtype)
                            stale_weight = _compute_footprint_overlap_ratio(
                                stale_mask_tensor,
                                stale_center_uv,
                                stale_radii,
                                stale_conics,
                                self.W,
                                self.H,
                            )
                        else:
                            stale_weight = _sample_mask_weights(
                                stale_mask_tensor,
                                stale_center_uv,
                                self.W,
                                self.H,
                            )
                        stale_mask_hit = stale_weight > 0.05
                        if stale_mask_hit.any():
                            if stale_opacity_hard_cap < 1.0:
                                stale_indices = torch.arange(
                                    old_point_num, device=self.device
                                )[stale_valid][stale_mask_hit]
                                opacity_hard_cap = self._merge_opacity_hard_cap(
                                    opacity_hard_cap,
                                    stale_indices,
                                    stale_opacity_hard_cap,
                                )
                                progress_dict["hard_stale_pts"] = str(int(stale_indices.numel()))
                            stale_weight = stale_weight[stale_mask_hit]
                            stale_opacity = stale_opacity[stale_mask_hit]
                            # Suppress stale splats based on conic-weighted footprint overlap ratio
                            # instead of a hard center-hit / max-hit heuristic.
                            loss_stale_opacity = (
                                stale_opacity.square() * stale_weight
                            ).sum() / stale_weight.sum().clamp_min(1e-6)
                            if stale_warmup_iters <= 1:
                                stale_warmup = 1.0
                            else:
                                stale_warmup = min(1.0, iteration / float(stale_warmup_iters - 1))
                            loss += lambda_stale_opacity * stale_warmup * loss_stale_opacity
                            progress_dict["stale_opacity"] = f"{loss_stale_opacity.item():.6f}"
                            progress_dict["stale_pts"] = str(int(stale_mask_hit.sum().item()))

            if (not camera_only and orphan_move_opacity_hard_cap < 1.0
                    and relabel_move_mask_tensor is not None
                    and hasattr(self, "last_still_mask")
                    and hasattr(self, "last_num")):
                old_point_num = min(self.last_num, self.last_still_mask.shape[0], uv.shape[0])
                if old_point_num > 0:
                    old_uv = uv[:old_point_num]
                    old_visible = (
                        (old_uv[:, 0] >= 0) & (old_uv[:, 0] <= self.W - 1) &
                        (old_uv[:, 1] >= 0) & (old_uv[:, 1] <= self.H - 1)
                    )
                    old_moving = ~self.last_still_mask[:old_point_num]
                    if (old_visible & old_moving).any():
                        relabel_support = _sample_mask_weights(
                            relabel_move_mask_tensor,
                            old_uv,
                            self.W,
                            self.H,
                        ) > 0.05
                        orphan_move = old_visible & old_moving & ~relabel_support
                        if orphan_move.any():
                            orphan_indices = torch.arange(
                                old_point_num, device=self.device
                            )[orphan_move]
                            opacity_hard_cap = self._merge_opacity_hard_cap(
                                opacity_hard_cap,
                                orphan_indices,
                                orphan_move_opacity_hard_cap,
                            )
                            progress_dict["orphan_move_pts"] = str(int(orphan_move.sum().item()))

            # === 2DGS 特有损失 ===
            if self.use_2dgs and not camera_only:
                # 缓存表面法线图，供后续致密化使用
                if return_dict.get("surf_normal") is not None:
                    self._cached_surf_normal = return_dict["surf_normal"].detach()  # (H,W,3)

                # 法线一致性损失：只在 iterations >= 20 且超过 normal_start_frac 时启用
                normal_start_iter = int(iterations * self.normal_start_frac)
                if (self.w_normal_consistency > 0 and iterations >= 20
                        and iteration >= normal_start_iter
                        and return_dict.get("rend_normal") is not None
                        and return_dict.get("surf_normal") is not None):
                    rend_normal = return_dict["rend_normal"]   # (H,W,3)
                    surf_normal = return_dict["surf_normal"]   # (H,W,3)
                    # alpha mask：用 detach 的 alpha 加权 surf_normal
                    if "alpha" in return_dict and return_dict["alpha"] is not None:
                        alpha_2d = return_dict["alpha"].squeeze(0).unsqueeze(-1)  # (H,W,1)
                        surf_normal = surf_normal * alpha_2d.detach()
                    normal_error = 1.0 - (rend_normal * surf_normal).sum(dim=-1)
                    loss_normal = normal_error.mean()
                    loss += self.w_normal_consistency * loss_normal
                    progress_dict["normal"] = f"{loss_normal.item():.6f}"

                # 深度失真损失：只在 iterations >= 20 且超过 dist_start_frac 时启用
                dist_start_iter = int(iterations * self.dist_start_frac)
                if (self.w_distortion > 0 and iterations >= 20
                        and iteration >= dist_start_iter
                        and return_dict.get("distort") is not None):
                    loss_distort = return_dict["distort"].mean()
                    loss += self.w_distortion * loss_distort
                    progress_dict["distort"] = f"{loss_distort.item():.6f}"

            if lambda_flow and hasattr(self, "gt_flow"): # ensure local flow consistency
                and_mask = (self.last_uv[:,0] > 0) & (self.last_uv[:,0] < self.W-1) & (self.last_uv[:,1] > 0) & (self.last_uv[:,1] < self.H-1)
                if hasattr(self, 'still_mask'):
                    if camera_only: # only optimize the still part
                        and_mask[:self.still_mask.shape[0]] = self.still_mask & and_mask[:self.still_mask.shape[0]]
                    else: # only optimize the move part
                        and_mask[:self.still_mask.shape[0]] = ~self.still_mask & and_mask[:self.still_mask.shape[0]]
                and_mask = and_mask.detach()

                pred_flow = uv[:self.last_num][and_mask] - self.last_uv[and_mask]

                y_coords_last = self.last_uv[and_mask][:,1].long()
                x_coords_last = self.last_uv[and_mask][:,0].long()

                gt_flow = self.gt_flow[y_coords_last, x_coords_last]


                loss_flow = mse_loss(pred_flow, gt_flow)
                loss += lambda_flow * loss_flow
                progress_dict["flow"] = f"{loss_flow.item():.6f}"

            self.optimizer.zero_grad()
            loss.backward()

            """gradient control"""
            # control all gaussians's gradients to be zero: rgb
            # 2DGS 模式下不冻结 RGB 梯度：gsplat 渲染语义与 msplat 不同，
            # 冻结颜色会导致灰色色块
            if hasattr(self, 'last_xyz'): # second frame and after
                for name, param in self._attributes.items():
                    if name == "rgb" and param.grad is not None:
                        if not self.use_2dgs:
                            param.grad = 0. * param.grad
            
            # control still gaussians's gradients to be zero: all
            if hasattr(self, 'still_mask'): # second frame and after
                for name, param in self._attributes.items():
                    if name == "xyz" and param.grad is not None:
                        param.grad[:self.still_mask.shape[0]][self.still_mask] = 0. * param.grad[:self.still_mask.shape[0]][self.still_mask]

            if camera_only:
                for name, param in self._attributes.items():
                    if param.grad is not None:
                        param.grad = 0. * param.grad

            """update the parameters"""
            self.optimizer.step()
            self._apply_opacity_hard_cap(opacity_hard_cap)
            if not self._freeze_scheduler_after_densify:
                self.scheduler.step()
            progress_dict["total"] = loss.item()
            now = time.time()
            finished_iters = iteration + 1
            remaining_iters = max(iterations - finished_iters, 0)
            avg_iter_time = (now - train_start_time) / finished_iters
            progress_dict["eta"] = _format_duration(avg_iter_time * remaining_iters)
            if global_total_steps is not None and global_start_time is not None:
                global_finished_steps = global_step_offset + finished_iters
                global_remaining_steps = max(global_total_steps - global_finished_steps, 0)
                global_avg_step_time = (now - global_start_time) / global_finished_steps
                progress_dict["all_eta"] = _format_duration(global_avg_step_time * global_remaining_steps)
            progress_bar.set_postfix(progress_dict)
            progress_bar.update(1)

            """densification"""
            # desify the occluded part
            if not camera_only and iteration == 0 and hasattr(self, 'last_xyz'): # second frame and after
                if mask.sum() > 0: # if there exists occluded points
                    self.densify_by_pixels(torch.ones_like(loss_rgb_pixel), error_threshold=0., percent=densify_occ_percent, mask=mask)

            # densify the detail-lacking part
            if not camera_only and densify_interval and (iteration+1) % densify_interval == 0 and (iteration+1) // densify_interval <= densify_times:
                if not hasattr(self, 'last_xyz'): # the first frame
                    self.densify_by_pixels(loss_rgb_pixel, error_threshold=densify_err_thre, percent=densify_err_percent, mask=None)
                else:
                    self.densify_by_pixels(loss_rgb_pixel, error_threshold=densify_err_thre, percent=densify_err_percent, mask=None) # [ ] TODO check densify hyperparameters

            if iteration % 10 == 0:
                # rgb
                rendered_rgb_np = render.render2img(rendered_rgb)
                frames.append(rendered_rgb_np)
                if self.use_2dgs and self.low_memory_2dgs:
                    rendered_depth_map_color = self._depth_map_color_from_points(uv.detach(), depth.detach())
                    rendered_center = rendered_rgb.detach()
                elif self.use_2dgs:
                    preview_input_group = [
                        item.detach() if torch.is_tensor(item) else item
                        for item in input_group
                    ]
                    with torch.no_grad():
                        preview_return_dict = self._render_backend(
                            preview_input_group,
                            ["depth_map_color", "center"],
                        )
                    rendered_depth_map_color = preview_return_dict["depth_map_color"]
                    rendered_center = preview_return_dict["center"]
                # depth map
                rendered_depth_map_color_np = render.render2img(rendered_depth_map_color)
                frames_depth.append(rendered_depth_map_color_np)
                # center
                rendered_center_np = render.render2img(rendered_center)
                frames_center.append(rendered_center_np)
        
        progress_bar.close()
        # import pdb; pdb.set_trace()

        final_return_type = ["rgb", "uv", "depth"]
        if self.use_2dgs and self.low_memory_2dgs:
            pass
        else:
            final_return_type.extend(["depth_map", "depth_map_color", "center"])
        if self.use_2dgs and not camera_only and not self.low_memory_2dgs:
            final_return_type.append("surf_normal")
        if save_geometry_eval:
            # Force the metric-only channels even for low-memory 2DGS.  This
            # render happens after optimization and is excluded from training.
            final_return_type.extend(["depth_expected", "alpha"])
        with torch.no_grad():
            final_return_dict = self._render_backend(
                self._make_input_group(),
                final_return_type,
            )
        if save_geometry_eval:
            self.save_geometry_eval(ckpt_name, final_return_dict)
        rendered_rgb = final_return_dict["rgb"]
        uv = final_return_dict["uv"]
        depth = final_return_dict["depth"]
        rendered_depth_map = final_return_dict.get("depth_map")
        if rendered_depth_map is None:
            rendered_depth_map = self.gt_depth.permute(2, 0, 1) if self.gt_depth is not None else torch.zeros((1, self.H, self.W), dtype=depth.dtype, device=depth.device)
        rendered_depth_map_color = final_return_dict.get("depth_map_color")
        if rendered_depth_map_color is None:
            rendered_depth_map_color = self._depth_map_color_from_points(uv, depth)
        rendered_center = final_return_dict.get("center")
        if rendered_center is None:
            rendered_center = rendered_rgb
        rendered_rgb_np = render.render2img(rendered_rgb)
        rendered_depth_map_color_np = render.render2img(rendered_depth_map_color)
        rendered_center_np = render.render2img(rendered_center)
        normal_map = final_return_dict.get("surf_normal")
        if normal_map is None:
            normal_source = self.gt_depth if self.gt_depth is not None else rendered_depth_map
            normal_map = self._build_depth_normal_map(normal_source)
        rendered_normal_np = self._normal_map_to_img(normal_map, rendered_depth_map)
        if frames:
            frames[-1] = rendered_rgb_np
            frames_depth[-1] = rendered_depth_map_color_np
            frames_center[-1] = rendered_center_np
        else:
            frames.append(rendered_rgb_np)
            frames_depth.append(rendered_depth_map_color_np)
            frames_center.append(rendered_center_np)
        
        """post-update prosessing"""
        if not camera_only:
            #### update still_mask
            within_mask = (uv[...,0] > 0) & (uv[...,0] < self.W-1) & (uv[...,1] > 0) & (uv[...,1] < self.H-1) # (N,)
            y_coords = uv[within_mask][:,1].long()
            x_coords = uv[within_mask][:,0].long()
            labels = ~move_mask[y_coords, x_coords] # (N_within,)

            self.still_mask = torch.ones(self.current_pts_num(), dtype=torch.bool).to(self.device).requires_grad_(False) # (N,)
            self.still_mask[within_mask] = labels
            self.still_mask_tentative = self.still_mask.detach().clone()
            if hasattr(self, 'last_still_mask'):
                old_point_num = min(self.last_still_mask.shape[0], self.still_mask.shape[0], within_mask.shape[0])
                old_within_mask = within_mask[:old_point_num]
                old_still_mask = self.still_mask[:old_point_num].clone()
                if old_within_mask.any() and relabel_move_mask is not None:
                    old_uv = uv[:old_point_num][old_within_mask]
                    old_y = old_uv[:, 1].long()
                    old_x = old_uv[:, 0].long()
                    # Use a wider relabel mask for visible old points so motion contours
                    # are less likely to get stuck as still due to tiny open-mask support.
                    old_still_mask[old_within_mask] = ~relabel_move_mask[old_y, old_x]
                # Only preserve historical labels for old points that are currently off-screen.
                # If an old point is visible in the current frame, allow move/still relabeling.
                old_still_mask[~old_within_mask] = self.last_still_mask[:old_point_num][~old_within_mask]
                self.still_mask[:old_point_num] = old_still_mask

            # print mask ratio
            print("\t[still] mask ratio is", self.still_mask.sum().item() / self.still_mask.size(0))

            self.move_seg = np.zeros((self.H, self.W), dtype=np.uint8)
            self.move_seg_erode = np.zeros((self.H, self.W), dtype=np.uint8)
            
            if uv[within_mask & ~self.still_mask].size(0) > 5:
                moving_seg_cluster = utils.FastConcaveHull2D(uv[within_mask & ~self.still_mask])
                ### get the convex hull segmentation
                self.move_seg = moving_seg_cluster.mask(self.W, self.H)
                self.move_seg = (self.move_seg * 255).astype(np.uint8)
                self.move_seg_erode = cv2.erode(self.move_seg, np.ones((20,20),np.uint8), iterations = 1)

            if hasattr(self, 'mask_prompt_pts'):
                propagate_uv = uv[:self.mask_prompt_pts.shape[0]][self.mask_prompt_pts]
                propagate_uv_within = (propagate_uv[:,0] > 0) & (propagate_uv[:,0] < self.W-1) & (propagate_uv[:,1] > 0) & (propagate_uv[:,1] < self.H-1)
                propagate_uv = propagate_uv[propagate_uv_within]
                if propagate_uv.size(0) > 4:
                    propagate_seg_cluster = utils.FastConcaveHull2D(propagate_uv)
                    self.propagate_seg = propagate_seg_cluster.mask(self.W, self.H)
                    self.propagate_seg = (self.propagate_seg * 255).astype(np.uint8)
                
            ### save current variables as last_variables for future use
            self.last_still_mask = self.still_mask.detach()
            self.last_uv = uv.detach()
            self.last_depth = depth.detach()
            self.last_xyz = self.get_attribute("xyz").detach()
            self.last_num = self.last_xyz.shape[0]
        
        ### render still points and moving points
        still_rgb_np = None
        still_center_np = None
        move_rgb_np = None
        move_center_np = None
        if hasattr(self, 'still_mask') and should_render_split_diagnostics(
            self.use_2dgs,
            self.force_split_diagnostics,
        ):
            # render still points
            input_group = [
                self.get_attribute("xyz")[:self.still_mask.shape[0]][self.still_mask],
                self.get_attribute("scale")[:self.still_mask.shape[0]][self.still_mask],
                self.get_attribute("rotate")[:self.still_mask.shape[0]][self.still_mask],
                self.get_attribute("opacity")[:self.still_mask.shape[0]][self.still_mask],
                self.get_attribute("rgb")[:self.still_mask.shape[0]][self.still_mask],
                self.intr,
                # self.extr,
                self.get_extr(),
                self.bg,
                self.W,
                self.H,
            ]

            return_dict = self._render_backend(
                input_group,
                ["rgb", "center"]
            )

            still_rgb_np = render.render2img(return_dict["rgb"])
            still_center_np = render.render2img(return_dict["center"])

            # render moving points
            input_group = [
                self.get_attribute("xyz")[:self.still_mask.shape[0]][~self.still_mask],
                self.get_attribute("scale")[:self.still_mask.shape[0]][~self.still_mask],
                self.get_attribute("rotate")[:self.still_mask.shape[0]][~self.still_mask],
                self.get_attribute("opacity")[:self.still_mask.shape[0]][~self.still_mask],
                self.get_attribute("rgb")[:self.still_mask.shape[0]][~self.still_mask],
                self.intr,
                # self.extr,
                self.get_extr(),
                self.bg,
                self.W,
                self.H,
            ]

            return_dict = self._render_backend(
                input_group,
                ["rgb", "center"]
            )

            move_rgb_np = render.render2img(return_dict["rgb"])
            move_center_np = render.render2img(return_dict["center"])

        if save_imgs:
            os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_{ckpt_name}.png"), rendered_rgb_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_center_{ckpt_name}.png"), rendered_center_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_depth_{ckpt_name}.png"), rendered_depth_map_color_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_normal_{ckpt_name}.png"), rendered_normal_np)
            if still_rgb_np is not None:
                imageio.imwrite(os.path.join(self.dir, "images", f"img_still_{ckpt_name}.png"), still_rgb_np)
                imageio.imwrite(os.path.join(self.dir, "images", f"img_still_center_{ckpt_name}.png"), still_center_np)
                imageio.imwrite(os.path.join(self.dir, "images", f"img_move_{ckpt_name}.png"), move_rgb_np)
                imageio.imwrite(os.path.join(self.dir, "images", f"img_move_center_{ckpt_name}.png"), move_center_np)
            if hasattr(self, 'move_seg'):
                os.makedirs(os.path.join(self.dir, "images_seg"), exist_ok=True)
                imageio.imwrite(os.path.join(self.dir, "images_seg", f"move_mask_{ckpt_name}.png"), self.move_seg)
            if hasattr(self, 'move_seg_erode'):
                os.makedirs(os.path.join(self.dir, "images_seg"), exist_ok=True)
                imageio.imwrite(os.path.join(self.dir, "images_seg", f"move_mask_erode_{ckpt_name}.png"), self.move_seg_erode)
            if hasattr(self, 'propagate_seg'):
                os.makedirs(os.path.join(self.dir, "images_seg"), exist_ok=True)
                imageio.imwrite(os.path.join(self.dir, "images_seg", f"propagate_mask_{ckpt_name}.png"), self.propagate_seg)

        if save_videos:
            # save them as a video with imageio
            frames_np = np.stack(frames, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_rgb.mp4"), frames_np, fps=30)
            frames_center_np = np.stack(frames_center, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_center.mp4"), frames_center_np, fps=30)
            frames_depth_np = np.stack(frames_depth, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_depth.mp4"), frames_depth_np, fps=30)

        if save_ckpt:
            self.save_checkpoint(ckpt_name=ckpt_name, camera_only=camera_only)

        return frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np, self.move_seg

    def eval(self, traj_index=None, line_scale=0.1, point_scale=0.3, alpha=0.5, split_interval=None):
        # line_scale = 0.6
        # point_scale = 2.
        # alpha = 0.6 # changing opacity
        num_traj = len(traj_index)
        # check if this class has a attribute traj_points
        if not hasattr(self, 'traj_xyz'): # the first frame
            self.traj_xyz = self.get_attribute("xyz")[traj_index].to(self.device).float()
            self.traj_scale = torch.ones((num_traj, 3), device=self.device).float()
            # create a no rotation quaternion
            self.traj_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(num_traj, 1).float()

            self.traj_opacity = self._activations_inv["opacity"](0.99*torch.ones((num_traj, 1), device=self.device).float())

            if split_interval is None or num_traj==split_interval:
                traj_rgb = torch.arange(0, 1, 1/num_traj, device=self.device).float().unsqueeze(1)
            else:
                traj_rgb_still = torch.arange(0, 1, 1/split_interval, device=self.device).float().unsqueeze(1)
                traj_rgb_move = torch.arange(0, 1, 1/(num_traj-split_interval), device=self.device).float().unsqueeze(1)
                traj_rgb = torch.cat([traj_rgb_still, traj_rgb_move], dim=0)
            traj_rgb = utils.apply_float_colormap(traj_rgb, colormap="gist_rainbow")
            self.traj_rgb = self._activations_inv["rgb"](traj_rgb)

            self.last_traj_xyz = self.traj_xyz
            self.last_traj_rgb = self.traj_rgb
        else: # the following frames
            current_xyz = self.get_attribute("xyz")[traj_index].to(self.device).float()
            line_xyz, line_rgb = utils.gen_line_set(self.last_traj_xyz, current_xyz, self.last_traj_rgb, device=self.device)
            num_in_line = line_xyz.shape[0]
            self.traj_xyz = torch.cat([self.traj_xyz, line_xyz], dim=0)
            num_total = self.traj_xyz.shape[0]
            # print(num_total)

            self.traj_scale = torch.ones((num_total, 3), device=self.device).float() * 1e-6 # too big will cause memory overflow

            self.traj_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(num_total, 1).float()

            # gradually fade out the opacity
            self.traj_opacity *= alpha
            # traj_opacity = torch.ones((num_in_line, 1), device=self.device)
            traj_opacity = self._activations_inv["opacity"](0.99*torch.ones((num_in_line, 1), device=self.device))
            self.traj_opacity = torch.cat([self.traj_opacity, traj_opacity], dim=0)
            # self.point_opacity = torch.ones((num_traj, 1), device=self.device)

            self.traj_rgb = torch.cat([self.traj_rgb, line_rgb], dim=0)
            # self.point_rgb = self.last_traj_rgb

            self.last_traj_xyz = self.get_attribute("xyz")[traj_index].to(self.device).float()
        input_group = [
            self.get_attribute("xyz"),
            self.get_attribute("scale"),
            self.get_attribute("rotate"),
            self.get_attribute("opacity"),
            self.get_attribute("rgb"),
            # self.intr, self.extr,
            self.intr, self.get_extr(),
            self.bg, self.W, self.H,
        ]

        output_dict = self._render_backend(
            input_group,
            ["rgb", "center", "depth_map_color"]
        )

        out_img = render.render2img(output_dict["rgb"])
        out_img_center = render.render2img(output_dict["center"])
        out_img_depth = render.render2img(output_dict["depth_map_color"])
        
        traj_group = [
            self.traj_xyz,
            self.traj_scale,
            self.traj_rotate,
            self.traj_opacity,
            self.traj_rgb,
            # self.intr, self.extr,
            self.intr, self.get_extr(),
            self.bg, self.W, self.H,
        ]

        out_traj = render.render_traj(
            traj_group, num_traj, line_scale, point_scale
        )

        out_img_traj = render.render2img(out_traj)

        # screen blending mode
        arr1 = np.array(out_img) / 255.0
        arr2 = np.array(out_img_traj) / 255.0

        # 应用滤色混色模式
        result = 1 - (1 - arr1) * (1 - arr2)

        # 将结果转换回 0-255 范围并转换为整数
        out_img_traj_upon = (result * 255).astype(np.uint8)
        # import pdb
        # pdb.set_trace()


        return out_img, out_img_center, out_img_depth, out_img_traj, out_img_traj_upon

    def render(self, xyz, scale, rotate, opacity, rgb):
        # 修复：移除了原来重复的 self.extr，严格保证 input_group 是 10 个元素
        input_group = [
            xyz,
            scale,
            rotate,
            opacity,
            rgb,
            self.intr,
            self.get_extr(),
            self.bg,
            self.W,
            self.H,
        ]

        output_dict = self._render_backend(
            input_group,
            ["rgb", "center", "depth_map_color"]
        )
        
        out_img = render.render2img(output_dict["rgb"])
        out_img_center = render.render2img(output_dict["center"])
        out_img_depth = render.render2img(output_dict["depth_map_color"])
        
        return out_img, out_img_center, out_img_depth


    def _prune_optimizer(self, mask, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict.keys():
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))).contiguous()
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True)).contiguous()
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, tensors_dict=None):
        if tensors_dict is None:
            tensors_dict = {
                "means": self.means,
                "scales": self.scales,
                "quats": self.quats,
                "rgbs": self.rgbs,
                "opacities": self.opacities,
            }
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, tensors_dict)
        self.means = optimizable_tensors["means"]
        self.scales = optimizable_tensors["scales"]
        self.quats = optimizable_tensors["quats"]
        self.rgbs = optimizable_tensors["rgbs"]
        self.opacities = optimizable_tensors["opacities"]

    def densify_by_pixels(self, error_map, error_threshold=1e-3, percent=0.1, mask=None, ):
        num_before = self.get_attribute("xyz").shape[0]
        error_map = error_map.detach().cpu().numpy()
        # print(error_map.shape)
        print("\n\t[densify] error_map's max, min, mean:", error_map.max().item(), error_map.min().item(), error_map.mean().item())
        # add uniform magnitude, to avoid zero probability
        positive_error = error_map[error_map > 0]
        error_map = error_map + (np.nanmin(positive_error) if positive_error.size else 1e-12)

        if mask is None:
            mask = (error_map > error_threshold).squeeze()
            print(f"\t[densify] No specify mask, generating it by error threshold: {error_threshold}")
        else:
            mask = mask.detach().cpu().numpy().squeeze()

        # transfer to bool type, then apply the mask
        mask = mask > 0
        error_map = error_map*mask[:,:error_map.shape[1]]

        mask_ratio = np.sum(mask) / np.size(mask)

        # make positive elements to be 1, and negative elements to be 0
        print("\t[densify] mask_ratio:", mask_ratio)
        densify_num = int(self.num_points * mask_ratio * percent)
        error_sum = np.sum(error_map)

        if densify_num > 0 and error_sum > 0:
            probability_distribution = error_map / error_sum
            # sample points from the probability distribution
            sampled_points = np.random.choice(a=np.arange(self.H*self.W), size=densify_num, p=probability_distribution.flatten())

            # convert the sampled points to coordinates on the image
            sampled_coordinates = np.unravel_index(sampled_points, (self.H, self.W))

            xys = np.array(sampled_coordinates).T[:,::-1].copy()
            depths = self.gt_depth[sampled_coordinates]
            scales = 1 / probability_distribution[sampled_coordinates]
            scales = np.ones_like(scales) * (1/self.num_points)

            scales = scales * (depths.cpu().numpy()/depths.cpu().numpy().min()).squeeze()
            rgbs = self.gt_image[sampled_coordinates]


            xys = torch.from_numpy(xys).to(self.device).float()
            depths = depths.to(self.device).float()

            new_xyz = geometry.pix2world(xys, depths, self.intr, self.get_extr())
            scales = torch.from_numpy(scales).float().unsqueeze(1).repeat(1, 3).to(self.device)
            scales = torch.clamp(scales, max=1e-3)
            if self.use_2dgs:
                scales = self._make_2dgs_scales(scales)


            new_scale = self._activations_inv["scale"](scales)
            rgbs = rgbs.contiguous()
            eps = 1e-6  # float32-safe clamp for torch.logit
            rgbs = torch.clamp(rgbs, min=eps, max=1-eps)
            # calculate the inverse of sigmoid function, i.e., logit function
            new_rgb = self._activations_inv["rgb"](rgbs)
            # 2DGS 模式：根据表面法线构造切平面朝向的 quaternion
            if self.use_2dgs and hasattr(self, '_cached_surf_normal'):
                new_rotate = self._build_tangent_quats(sampled_coordinates)
            else:
                new_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(new_xyz.shape[0], 1).float()
            new_opacity = 0.99*torch.ones((new_xyz.shape[0], 1), device=self.device).float()
            new_opacity = self._activations_inv["opacity"](new_opacity)

            self.densification_postfix(new_xyz, new_scale, new_rotate, new_opacity, new_rgb)

        num_after = self.get_attribute("xyz").shape[0]
        print(f"\t[densify / split] number of gaussians: {num_before} -> {num_after}")

    def _build_tangent_quats(self, sampled_coordinates):
        """根据缓存的表面法线图，为采样点构造切平面朝向的 quaternion。

        核心思路：
        1. 从 surf_normal map 读取每个采样点的法线 n
        2. 通过 Gram-Schmidt 正交化构造旋转矩阵 R = [t1, t2, n]
        3. 用 roma 将旋转矩阵转为 unitquat

        参数:
            sampled_coordinates: tuple (y_indices, x_indices)，像素坐标

        返回:
            quats: (N, 4) 四元数
        """
        normals = self._cached_surf_normal[sampled_coordinates]  # (N, 3)
        return self._quats_from_normals(normals)

    def _quats_from_normals(self, normals):
        normals = torch.nn.functional.normalize(normals, dim=-1)
        fallback = (-self._camera_forward_world()).expand_as(normals)
        valid = torch.isfinite(normals).all(dim=-1) & (normals.norm(dim=-1) > 0.5)
        normals = torch.where(valid.unsqueeze(-1), normals, fallback)

        up = torch.tensor([0., 1., 0.], device=normals.device).expand_as(normals)
        parallel_mask = (normals * up).sum(dim=-1).abs() > 0.99
        up_alt = torch.tensor([1., 0., 0.], device=normals.device).expand_as(normals)
        up = torch.where(parallel_mask.unsqueeze(-1), up_alt, up)

        t1 = torch.cross(up, normals, dim=-1)
        t1 = torch.nn.functional.normalize(t1, dim=-1)
        t2 = torch.cross(normals, t1, dim=-1)
        t2 = torch.nn.functional.normalize(t2, dim=-1)
        R = torch.stack([t1, t2, normals], dim=-1)
        return roma.rotmat_to_unitquat(R)

    def _make_2dgs_scales(self, scales):
        scales = scales.clone()
        normal_scale = scales[:, :2].mean(dim=1) * 0.01
        scales[:, 2] = normal_scale.clamp(min=1e-6, max=1e-5)
        return scales

    def _depth_map_color_from_points(self, uv, depth):
        depth_values = depth.squeeze(-1)
        depth_map = torch.zeros((self.H, self.W), dtype=depth.dtype, device=depth.device)
        valid = (
            (uv[:, 0] >= 0) & (uv[:, 0] < self.W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < self.H) &
            (depth_values > 0)
        )
        if valid.any():
            x = uv[valid, 0].long().clamp(0, self.W - 1)
            y = uv[valid, 1].long().clamp(0, self.H - 1)
            depth_map[y, x] = depth_values[valid]

        valid_depth = depth_map > 0
        if valid_depth.any():
            normalized = depth_map - depth_map[valid_depth].min()
            normalized = normalized / (normalized.max() + 1e-5)
        else:
            normalized = depth_map
        normalized = torch.clamp(torch.nan_to_num(normalized, nan=0.0), 0.0, 1.0)
        return normalized.unsqueeze(0).expand(3, -1, -1).contiguous()

    def _camera_forward_world(self):
        extr = self.get_extr()
        bottom = torch.tensor([[0., 0., 0., 1.]], dtype=extr.dtype, device=extr.device)
        cam2world = torch.linalg.inv(torch.cat([extr, bottom], dim=0))
        return torch.nn.functional.normalize(cam2world[:3, 2], dim=0)

    def _build_depth_normal_map(self, gt_depth):
        depth = gt_depth.squeeze(-1) if gt_depth.dim() == 3 else gt_depth
        ys, xs = torch.meshgrid(
            torch.arange(self.H, device=self.device, dtype=depth.dtype),
            torch.arange(self.W, device=self.device, dtype=depth.dtype),
            indexing="ij",
        )
        uv = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
        pts = geometry.pix2world(
            uv,
            depth.reshape(-1, 1),
            self.intr,
            self.get_extr(),
        ).reshape(self.H, self.W, 3)

        dx = torch.zeros_like(pts)
        dy = torch.zeros_like(pts)
        dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
        dx[:, 0] = pts[:, 1] - pts[:, 0]
        dx[:, -1] = pts[:, -1] - pts[:, -2]
        dy[1:-1] = pts[2:] - pts[:-2]
        dy[0] = pts[1] - pts[0]
        dy[-1] = pts[-1] - pts[-2]

        normals = torch.cross(dx, dy, dim=-1)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        extr = self.get_extr()
        bottom = torch.tensor([[0., 0., 0., 1.]], dtype=extr.dtype, device=extr.device)
        cam2world = torch.linalg.inv(torch.cat([extr, bottom], dim=0))
        camera_center = cam2world[:3, 3]
        view_dir = torch.nn.functional.normalize(camera_center - pts, dim=-1)
        normals = torch.where(
            (normals * view_dir).sum(dim=-1, keepdim=True) < 0,
            -normals,
            normals,
        )
        fallback = (-self._camera_forward_world()).view(1, 1, 3)
        valid = torch.isfinite(normals).all(dim=-1, keepdim=True) & (
            normals.norm(dim=-1, keepdim=True) > 0.5
        )
        return torch.where(valid, normals, fallback)

    def _normal_map_to_img(self, normal_map, depth_map=None):
        normals = normal_map
        if normals.dim() == 4:
            normals = normals.squeeze(0)
        if normals.dim() == 3 and normals.shape[0] == 3:
            normals = normals.permute(1, 2, 0)
        normals = torch.nn.functional.normalize(normals, dim=-1)
        valid = torch.isfinite(normals).all(dim=-1, keepdim=True)
        if depth_map is not None:
            depth = depth_map
            if depth.dim() == 3:
                depth = depth.squeeze(0).squeeze(-1)
            valid = valid & (depth.unsqueeze(-1) > 0)
        normal_img = torch.clamp(normals * 0.5 + 0.5, 0.0, 1.0)
        normal_img = torch.where(valid, normal_img, torch.zeros_like(normal_img))
        return (normal_img.detach().cpu().numpy() * 255).astype(np.uint8)

    def densification_postfix(self, new_xyz, new_scale, new_rotate, new_opacity, new_rgb):
        self._append_optimizer_tensor("xyz", new_xyz)
        self._append_optimizer_tensor("scale", new_scale)
        self._append_optimizer_tensor("rotate", new_rotate)
        self._append_optimizer_tensor("opacity", new_opacity)
        self._append_optimizer_tensor("rgb", new_rgb)
        if self.legacy_densify_lr_after_densify:
            for group in self.optimizer.param_groups:
                if group.get("name") == "attributes":
                    group["lr"] = self.lr
            self._freeze_scheduler_after_densify = True

    def project_points(self, points):
        # return msplat.project_point(points, self.intr, self.extr, self.W, self.H)
        return msplat.project_point(points, self.intr, self.get_extr(), self.W, self.H)
    
