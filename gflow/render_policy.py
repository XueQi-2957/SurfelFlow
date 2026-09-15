def _append_once(items, value):
    if value not in items:
        items.append(value)


def should_render_split_diagnostics(use_2dgs, force_split_diagnostics=False):
    return (not use_2dgs) or force_split_diagnostics


def train_render_return_types(
    use_2dgs,
    camera_only,
    iteration,
    iterations,
    w_normal_consistency,
    w_distortion,
    normal_start_frac,
    dist_start_frac,
    need_surf_normal_cache=False,
    low_memory_2dgs=False,
):
    if not use_2dgs:
        return ["rgb", "uv", "depth", "depth_map", "depth_map_color", "center"]

    if low_memory_2dgs:
        return ["rgb", "uv", "depth"]

    return_type = ["rgb", "uv", "depth", "depth_map"]
    if camera_only:
        return return_type

    if need_surf_normal_cache:
        _append_once(return_type, "surf_normal")

    normal_start_iter = int(iterations * normal_start_frac)
    normal_active = (
        w_normal_consistency > 0
        and iterations >= 20
        and iteration >= normal_start_iter
    )
    if normal_active:
        _append_once(return_type, "alpha")
        _append_once(return_type, "rend_normal")
        _append_once(return_type, "surf_normal")

    dist_start_iter = int(iterations * dist_start_frac)
    dist_active = (
        w_distortion > 0
        and iterations >= 20
        and iteration >= dist_start_iter
    )
    if dist_active:
        _append_once(return_type, "distort")

    return return_type
