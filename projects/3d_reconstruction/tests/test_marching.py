"""projects/3d_reconstruction/marching 的测试 —— 直接运行：``python tests/test_marching.py``。

验证 Marching Tetrahedra 提取（教程第 05 章）：单位球 SDF -> 网格平均半径偏差
< 0.05、闭合性（每条边恰被 2 个三角形共享）、法向朝外（比例 > 95%）、顶点
落在零水平集上、空场/全负场输出空网格、盒子提取的面法向与范围、三角形
取向与场梯度一致，以及 NaN（未观测）区域的权重掩码提取。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marching import MT_TRIANGLES, _TET_EDGES, marching_tetrahedra  # noqa: E402

# 共享网格参数：单位球，采样域 [-1.5, 1.5]^3，分辨率 40（voxel = 0.075 m）。
N = 40
LO = -1.5
VOXEL = 3.0 / N


def _sphere_field(radius=1.0):
    """单位球 SDF 的体素中心采样场 (N, N, N)。"""
    axis = LO + np.arange(N) * VOXEL
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.sqrt(xx ** 2 + yy ** 2 + zz ** 2) - radius


def _edge_counts(mesh):
    """无向边的共享计数数组（用于水密性检查）。"""
    tri = mesh.triangles
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    keys = keys[:, 0].astype(np.int64) * len(mesh.vertices) + keys[:, 1]
    _, counts = np.unique(keys, return_counts=True)
    return counts


def test_mt_table_matches_sign_configuration():
    """16 case 表自洽：每个 case 的三角形棱集合 = 该符号配置的变号棱集合。"""
    for case in range(16):
        inside = [(case >> k) & 1 for k in range(4)]
        crossed = {e for e, (a, b) in enumerate(_TET_EDGES) if inside[a] != inside[b]}
        tris = MT_TRIANGLES[case]
        used = {e for tri in tris for e in tri}
        n_expect = 0 if len(crossed) == 0 else (2 if len(crossed) == 4 else 1)
        assert used == crossed and len(tris) == n_expect, (
            case, sorted(crossed), tris)


def test_sphere_radius_deviation_below_0p05():
    """单位球 SDF -> 网格顶点平均半径偏差 < 0.05（教学实现的精度底线）。"""
    mesh = marching_tetrahedra(_sphere_field(), np.full(3, LO), VOXEL)
    r = np.linalg.norm(mesh.vertices, axis=1)
    mean_dev = float(np.abs(r - 1.0).mean())
    assert 0.2 < r.mean() < 2.0                      # 网格确实围出了球
    assert mean_dev < 0.05, mean_dev
    print(f"\n[mt] sphere: V={len(mesh.vertices)} T={len(mesh.triangles)} "
          f"mean|r-1|={mean_dev:.5f} max|r-1|={np.abs(r - 1.0).max():.4f}")


def test_mesh_is_closed_every_edge_shared_by_two_triangles():
    """闭合性：每条无向边恰被 2 个三角形共享（水密、边流形）。"""
    mesh = marching_tetrahedra(_sphere_field(), np.full(3, LO), VOXEL)
    counts = _edge_counts(mesh)
    assert (counts == 2).all(), f"非流形边: {np.unique(counts, return_counts=True)}"


def test_normals_point_outward():
    """法向朝外：n·(v - center) > 0 的顶点比例 > 95%（外正内负 (1.13)）。"""
    mesh = marching_tetrahedra(_sphere_field(), np.full(3, LO), VOXEL)
    nrm = mesh.vertex_normals / np.linalg.norm(mesh.vertex_normals, axis=1, keepdims=True)
    outward = np.einsum("ij,ij->i", nrm, mesh.vertices)  # center = 原点
    frac = float((outward > 0).mean())
    assert frac > 0.95, frac


def test_vertices_on_zero_level_set():
    """顶点落在零水平集上：|sdf(v)| 远小于一个体素（(5.1) 线性插值的直接后果）。"""
    field = _sphere_field()
    mesh = marching_tetrahedra(field, np.full(3, LO), VOXEL)
    # 顶点 SDF 用三线性插值近似读回：|sdf| <= 0.5 体素（线性插值 + 弯曲零点误差）。
    axis = LO + np.arange(N) * VOXEL
    g = (mesh.vertices - LO) / VOXEL
    g0 = np.clip(np.floor(g).astype(int), 0, N - 2)
    frac = np.clip(g, 0, N - 1) - g0
    acc = np.zeros(len(mesh.vertices))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = ((frac[:, 0] if dx else 1 - frac[:, 0])
                       * (frac[:, 1] if dy else 1 - frac[:, 1])
                       * (frac[:, 2] if dz else 1 - frac[:, 2]))
                acc += wgt * field[g0[:, 2] + dz, g0[:, 1] + dy, g0[:, 0] + dx]
    assert np.abs(acc).max() <= 0.5 * VOXEL, np.abs(acc).max()


def test_empty_positive_field_gives_empty_mesh():
    """空 SDF（全场为正，零水平集不在域内）-> 空网格 (0 顶点 / 0 三角形)。"""
    mesh = marching_tetrahedra(np.ones((12, 12, 12)), np.zeros(3), 1.0)
    assert mesh.vertices.shape == (0, 3)
    assert mesh.triangles.shape == (0, 3)
    assert mesh.vertex_normals.shape == (0, 3)


def test_fully_negative_field_gives_empty_mesh():
    """全负场（表面完全在域外）-> 空网格（case 15 恒空表）。"""
    mesh = marching_tetrahedra(-np.ones((12, 12, 12)) - 0.5, np.zeros(3), 1.0)
    assert len(mesh.triangles) == 0 and len(mesh.vertices) == 0


def test_box_extraction_extent_and_face_normals():
    """盒子提取：顶点范围 ≈ half（±0.5 体素）、面内法向轴对齐、网格闭合。"""
    half = np.array([0.6, 0.5, 0.4])
    axis = LO + np.arange(N) * VOXEL
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    q = np.stack([np.abs(xx) - half[0], np.abs(yy) - half[1], np.abs(zz) - half[2]],
                 axis=-1)
    field = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
    mesh = marching_tetrahedra(field, np.full(3, LO), VOXEL)
    lo_v, hi_v = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    assert np.all(np.abs(hi_v - half) <= 0.5 * VOXEL + 1e-9), hi_v
    assert np.all(np.abs(lo_v + half) <= 0.5 * VOXEL + 1e-9), lo_v
    # 面内部法向轴对齐（棱/角附近因 SDF 梯度不连续而偏斜属预期）。
    nrm = mesh.vertex_normals / np.linalg.norm(mesh.vertex_normals, axis=1, keepdims=True)
    axis_aligned = float((np.abs(nrm).max(axis=1) > 0.99).mean())
    assert axis_aligned > 0.6, axis_aligned
    assert (_edge_counts(mesh) == 2).all()
    print(f"[mt] box: V={len(mesh.vertices)} T={len(mesh.triangles)} "
          f"axis-aligned normals {axis_aligned:.3f}")


def test_triangle_orientation_follows_gradient():
    """取向统一：每个三角形的面法向与其质心处顶点法向（梯度方向）同侧。"""
    mesh = marching_tetrahedra(_sphere_field(), np.full(3, LO), VOXEL)
    v = mesh.vertices[mesh.triangles]
    face_n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    grad_ref = mesh.vertex_normals[mesh.triangles].mean(axis=1)
    dots = np.einsum("ij,ij->i", face_n, grad_ref)
    assert (dots > 0).all(), f"{(dots <= 0).sum()} 个三角形取向反了"


def test_nan_unknown_region_skipped():
    """权重掩码：未观测（NaN）区域被跳过——远端 NaN 化不改变剩余网格。"""
    field = _sphere_field()
    mesh_full = marching_tetrahedra(field, np.full(3, LO), VOXEL)
    # 把远离表面的正数区（|z| > 1.3 的切片）置 NaN：表面网格应完全不变。
    axis = LO + np.arange(N) * VOXEL
    far = np.abs(axis) > 1.3
    masked = field.copy()
    masked[far, :, :] = np.nan
    mesh_masked = marching_tetrahedra(masked, np.full(3, LO), VOXEL)
    assert len(mesh_masked.triangles) == len(mesh_full.triangles)
    assert np.allclose(mesh_masked.vertices, mesh_full.vertices, atol=1e-12)
    # NaN 切穿表面时：被切区域无几何，边界边（计数 1）恰落在切平面上。
    cut = field.copy()
    cut[:N // 2, :, :] = np.nan                      # 下半空间未观测
    mesh_cut = marching_tetrahedra(cut, np.full(3, LO), VOXEL)
    plane_z = LO + N // 2 * VOXEL                    # 切平面 z = 0
    assert not (mesh_cut.vertices[:, 2] < plane_z - 1e-9).any()
    tri = mesh_cut.triangles
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    uniq, cnt = np.unique(keys[:, 0].astype(np.int64) * len(mesh_cut.vertices)
                          + keys[:, 1], return_counts=True)
    assert np.isin(cnt, (1, 2)).all()                # 只有面边界与内部两类边
    boundary = uniq[cnt == 1]
    bverts = np.unique(np.stack([boundary // len(mesh_cut.vertices),
                                 boundary % len(mesh_cut.vertices)]).ravel())
    assert len(bverts) > 8                           # 边界构成一个圈
    assert np.allclose(mesh_cut.vertices[bverts, 2], plane_z, atol=1e-9)


if __name__ == "__main__":
    # 单点测试：python tests/test_X.py <测试名子串>；不带参数 = 全部测试。
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and pat in k]
    if not fns:
        print(f"没有匹配 '{pat}' 的测试；可用测试：")
        for k in sorted(globals()):
            if k.startswith("test_"):
                print(" ", k)
        sys.exit(1)
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
    sys.exit(1 if failed else 0)
