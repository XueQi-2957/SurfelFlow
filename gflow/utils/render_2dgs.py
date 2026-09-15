"""2DGS 渲染器封装模块

基于 gsplat.rendering.rasterization_2dgs，提供与原 msplat render_multiple 兼容的接口。
所有 gsplat 调用集中在此文件，不要散落到 trainer.py 各处。

硬约束：
- packed=False（第一版固定）
- sh_degree=0 + colors=[N,1,3]（规避 gsplat 1.5.3 的 shape 断言问题）
- extr 只补行为 4x4，不做 inverse（GFlow extr 已是 w2c）
- uv/depth 主链路走 msplat.project_point 兼容路径
"""

import os

import torch
import msplat

# Default to the user's common GPUs: RTX 3090 (8.6) and Ada 4060/4080/4090 (8.9).
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6;8.9")

from gsplat.rendering import rasterization_2dgs
from .color import apply_float_colormap
import numpy as np


# ============================================================
# A. 内参转换
# ============================================================
def make_Ks_from_intr(intr):
    """将 GFlow 的 intr (4,) 转换为 gsplat 的 Ks (1,3,3) 内参矩阵。

    输入: intr (4,) = [fx, fy, cx, cy]
    输出: Ks (1,3,3)
    不做求导截断，dtype/device 与输入一致。
    """
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    Ks = torch.zeros(1, 3, 3, dtype=intr.dtype, device=intr.device)
    Ks[0, 0, 0] = fx
    Ks[0, 1, 1] = fy
    Ks[0, 0, 2] = cx
    Ks[0, 1, 2] = cy
    Ks[0, 2, 2] = 1.0
    return Ks


# ============================================================
# B. 外参转换
# ============================================================
def make_viewmat_from_extr(extr):
    """将 GFlow 的 extr (3,4) world-to-camera 矩阵转换为 gsplat 的 viewmats (1,4,4)。

    只在最后补一行 [0,0,0,1]，禁止额外 inverse()。
    GFlow 的 extr 语义已经是 world-to-camera，直接用。

    输入: extr (3,4)
    输出: viewmats (1,4,4)
    """
    bottom = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]], dtype=extr.dtype, device=extr.device
    )
    viewmat = torch.cat([extr, bottom], dim=0)  # (4,4)
    return viewmat.unsqueeze(0)  # (1,4,4)


# ============================================================
# C. 兼容投影函数（主链路使用）
# ============================================================
def project_points_compat(xyz, intr, extr, W, H):
    """2DGS 模式下主训练链路的 uv/depth 来源。

    直接调用 msplat.project_point，保持与原 3DGS 路径完全一致的语义。
    第一版不要默认改成从 meta["means2d"] 取。

    输入:
        xyz: (N, 3)
        intr: (4,) = [fx, fy, cx, cy]
        extr: (3, 4) world-to-camera
        W, H: int
    输出:
        uv: (N, 2)
        depth: (N, 1)
    """
    (uv, depth) = msplat.project_point(xyz, intr, extr, W, H)
    return uv, depth


# ============================================================
# C'. 调试 helper（不在主链路中使用）
# ============================================================
def project_points_from_meta(meta):
    """从 gsplat meta 字典中提取 uv 和 depth。

    仅调试/对拍用途，默认不启用。
    返回: (means2d, depths) 或 (None, None)
    """
    if meta is None:
        return None, None
    means2d = meta.get("means2d", None)
    depths = meta.get("depths", None)
    if means2d is None or depths is None:
        return None, None
    # means2d: (1, N, 2) -> (N, 2)
    if means2d.dim() == 3:
        means2d = means2d.squeeze(0)
    # depths: (1, N) -> (N, 1)
    if depths.dim() == 2:
        depths = depths.squeeze(0)
    if depths.dim() == 1:
        depths = depths.unsqueeze(-1)
    return means2d, depths


# ============================================================
# D. Center 可视化渲染（专用 helper）
# ============================================================
def render_center_2dgs(
    xyz, rgb, intr, extr, bg, W, H,
    center_scale_world=5e-4, packed=False, sh_degree=0,
):
    """专门给 "center" 可视化使用的 2DGS 渲染函数。

    不复用学习到的 quats/scales/opacities，全部使用固定值。
    不回退 msplat，完全在 2DGS 管线内完成。

    固定参数:
        quats: canonical identity quaternion [1,0,0,0]
        scales: center_scale_world（默认 5e-4）
        opacities: 全 1

    输入:
        xyz: (N,3) 高斯中心坐标
        rgb: (N,3) 颜色
        intr: (4,) 内参
        extr: (3,4) 外参 (w2c)
        bg: float 背景色
        W, H: int 图像宽高
    输出:
        rendered_center: (3,H,W)
    """
    N = xyz.shape[0]
    device = xyz.device

    # 固定 canonical quaternion
    quats = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device, dtype=xyz.dtype
    ).unsqueeze(0).expand(N, -1).contiguous()  # (N,4)

    # 固定小 scale
    scales = torch.full(
        (N, 3), center_scale_world, device=device, dtype=xyz.dtype
    )  # (N,3)

    # 全 1 opacity
    opacities = torch.ones(N, device=device, dtype=xyz.dtype)  # (N,)

    # 颜色走 [N,1,3] + sh_degree=0 路径
    #由于 gsplat 1.5.3 存在 SH=0 的默认行为 (RGB = 0.28209 * SH_c0 + 0.5)
    # 此处进行逆向平移映射，使其等效于原始透传颜色
    SH_C0 = 0.28209479177387814
    colors = (rgb - 0.5) / SH_C0
    colors = colors.unsqueeze(1)  # (N,1,3)

    Ks = make_Ks_from_intr(intr)
    viewmats = make_viewmat_from_extr(extr)
    backgrounds = torch.full((1, 3), bg, dtype=xyz.dtype, device=device)

    render_colors, _, _, _, _, _, _ = rasterization_2dgs(
        means=xyz,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        render_mode="RGB",
        packed=packed,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
    )

    # (1,H,W,3) -> (3,H,W)
    rendered_center = render_colors.squeeze(0).permute(2, 0, 1)
    return rendered_center


# ============================================================
# E. 主渲染函数
# ============================================================
def render_multiple_2dgs(
    input_group,
    return_type=None,
    packed=False,
    sh_degree=0,
    center_scale_world=5e-4,
):
    """2DGS 渲染主函数，接口与原 render.render_multiple() 兼容。

    输入:
        input_group: [xyz, scale, rotate, opacity, rgb, intr, extr, bg, W, H]
            xyz: (N,3), scale: (N,3), rotate: (N,4), opacity: (N,1),
            rgb: (N,3), intr: (4,), extr: (3,4), bg: float, W: int, H: int

        return_type: list[str]，支持:
            "rgb", "uv", "depth", "depth_map", "depth_map_color",
            "center", "alpha", "rend_normal", "surf_normal", "distort"

    输出:
        return_dict: dict，包含请求的渲染结果

    硬约束:
        - packed=False 固定
        - sh_degree=0 固定
        - 颜色走 [N,1,3] 路径
        - uv/depth 走 msplat 兼容投影，不从 meta 取
        - extr 只补行不 inverse
    """
    if return_type is None:
        return_type = ["rgb"]

    xyz, scale, rotate, opacity, rgb, intr, extr, bg, W, H = input_group
    return_dict = {}

    # === uv / depth：走兼容投影路径（主链路） ===
    if "uv" in return_type or "depth" in return_type:
        uv, depth = project_points_compat(xyz, intr, extr, W, H)
        if "uv" in return_type:
            return_dict["uv"] = uv
        if "depth" in return_type:
            return_dict["depth"] = depth

    # === 判断需要的渲染特性 ===
    needs_depth = any(
        k in return_type
        for k in ["depth_map", "depth_map_color", "depth_expected", "surf_normal"]
    )
    needs_distort = "distort" in return_type
    needs_normal = "rend_normal" in return_type or "surf_normal" in return_type

    # 选择 render_mode：需要深度/法线时使用 RGB+ED
    if needs_depth or needs_normal:
        render_mode = "RGB+ED"
    else:
        render_mode = "RGB"

    # === 转换参数格式 ===
    Ks = make_Ks_from_intr(intr)
    viewmats = make_viewmat_from_extr(extr)

    # opacity: (N,1) -> (N,)
    opacities = opacity.squeeze(-1)

    # 颜色：第一版统一用 [N,1,3] + sh_degree=0
    #由于 gsplat 1.5.3 存在 SH=0 的默认行为 (RGB = 0.28209 * SH_c0 + 0.5)
    # 此处进行逆向平移映射，使其等效于原始透传颜色
    SH_C0 = 0.28209479177387814
    colors = (rgb - 0.5) / SH_C0
    colors = colors.unsqueeze(1)  # (N,1,3)

    # 背景色：gsplat 内部会在 RGB+ED 时自动拼接 depth 背景通道
    backgrounds = torch.full((1, 3), bg, dtype=xyz.dtype, device=xyz.device)

    # === 调用 gsplat 2DGS 渲染 ===
    (
        render_colors,      # (1,H,W,X) X=3(RGB) 或 4(RGB+ED)
        render_alphas,      # (1,H,W,1)
        render_normals,     # (1,H,W,3)，已从相机空间转到世界空间
        surf_normals,       # (H,W,3) 或 None（仅 RGB+ED 时有值）
        render_distort,     # (1,H,W,1) 或 None
        render_median,      # (1,H,W,1)
        meta,               # dict
    ) = rasterization_2dgs(
        means=xyz,
        quats=rotate,
        scales=scale,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        render_mode=render_mode,
        packed=packed,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        distloss=needs_distort,
        depth_mode="expected",
    )

    # === 输出格式转换 ===

    # RGB: (1,H,W,3+) -> (3,H,W)
    if "rgb" in return_type:
        rendered_rgb = render_colors[0, :, :, :3].permute(2, 0, 1)  # (3,H,W)
        return_dict["rgb"] = rendered_rgb

    # Alpha: (1,H,W,1) -> (1,H,W)
    if "alpha" in return_type:
        return_dict["alpha"] = render_alphas[0, :, :, 0].unsqueeze(0)  # (1,H,W)

    # Depth map: 从 RGB+ED 的最后一个通道提取
    if "depth_map" in return_type or "depth_map_color" in return_type or "depth_expected" in return_type:
        if render_mode == "RGB+ED":
            # expected depth 在最后一个通道，(1,H,W,1) -> (1,H,W)
            rendered_depth_map = render_colors[0, :, :, -1:].permute(2, 0, 1)  # (1,H,W)
        else:
            rendered_depth_map = torch.zeros(
                1, H, W, dtype=xyz.dtype, device=xyz.device
            )

        if "depth_map" in return_type:
            return_dict["depth_map"] = rendered_depth_map

        if "depth_expected" in return_type:
            # RGB+ED with depth_mode="expected" already returns the
            # opacity-normalized expected camera-z depth.
            return_dict["depth_expected"] = rendered_depth_map

        if "depth_map_color" in return_type:
            # 对渲染出的 depth map 用 colormap 上色
            # 注意：apply_float_colormap 内部 matplotlib 切片不支持 (H,W,1) 输入
            # 所以手动实现 depth colormap
            from matplotlib import cm as mpl_cm
            depth_2d = rendered_depth_map.squeeze(0)  # (H,W)
            d = depth_2d.detach()
            # non_zero 归一化：排除零值
            non_zero_mask = d != 0
            if non_zero_mask.any():
                d = d - d[non_zero_mask].min()
            d = d / (d.max() + 1e-5)
            d = torch.clamp(d, 0, 1)
            d_np = (d.cpu().numpy() * 255).astype(np.int64)
            # matplotlib turbo colormap，返回 (H,W,4)
            colored_np = mpl_cm.get_cmap("turbo")(d_np)[..., :3]  # (H,W,3)
            rendered_depth_map_color = torch.tensor(
                colored_np, device=xyz.device, dtype=xyz.dtype
            ).permute(2, 0, 1)  # (3,H,W)
            return_dict["depth_map_color"] = rendered_depth_map_color

    # Render normals: (1,H,W,3) -> (H,W,3)
    if "rend_normal" in return_type:
        if render_normals is not None:
            rn = render_normals
            if rn.dim() == 4:
                rn = rn.squeeze(0)  # (H,W,3)
            return_dict["rend_normal"] = rn
        else:
            return_dict["rend_normal"] = None

    # Surface normals (from depth): (H,W,3) 或 None
    if "surf_normal" in return_type:
        if surf_normals is not None:
            sn = surf_normals
            if sn.dim() == 4:
                sn = sn.squeeze(0)  # (H,W,3)
            return_dict["surf_normal"] = sn
        else:
            return_dict["surf_normal"] = None

    # Distortion: (1,H,W,1) -> (H,W,1)
    if "distort" in return_type:
        if render_distort is not None:
            dt = render_distort
            if dt.dim() == 4:
                dt = dt.squeeze(0)  # (H,W,1)
            return_dict["distort"] = dt
        else:
            return_dict["distort"] = None

    # Center 可视化：走专用 helper，不污染主渲染逻辑
    if "center" in return_type:
        rendered_center = render_center_2dgs(
            xyz, rgb, intr, extr, bg, W, H,
            center_scale_world=center_scale_world,
            packed=packed,
            sh_degree=sh_degree,
        )
        return_dict["center"] = rendered_center

    # Meta 原样保留供调试
    return_dict["meta"] = meta

    return return_dict
