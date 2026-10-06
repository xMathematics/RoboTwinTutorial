"""Marching Tetrahedra 等值面提取（marching）—— 教程第 05 章的教学实现。

函数流水线
----------
本模块承接 ``tsdf``（tsdf.py）融合出的体素标量场，产出三角网格：

    marching_tetrahedra(field, origin, voxel_size)
        立方体 -> 6 四面体拆分（共享体对角线，教程 5.1 ③(b)）
        -> 每四面体按 4 角点符号查 16 case 表（0/1/2 个三角形）
        -> 棱上交点线性插值 t = f_a/(f_a - f_b)（教程 (5.1)）
        -> 顶点去重（全局棱键）+ 顶点法向 = 梯度方向（教程 (5.2)）
        -> 三角形取向统一为"外法向"（与场梯度同向，回引 (1.3)/(1.13)）
        -> TriangleMesh(vertices, triangles, vertex_normals)

依赖方向：依赖 ``scene``（无——仅 numpy）；输入场通常来自 ``tsdf.TSDFVolume``
（注意其样本在体素中心，取 ``origin + 0.5·voxel`` 作为采样原点）；
``demo.py`` 与 ``tests/test_marching.py`` 直接驱动。

输入/输出：输入标量场 (n, n, n)（无量纲或 m，零等值面即表面，回引 (1.2)）、
采样原点 (3,)（m）、体素边长（m）；输出 :class:`TriangleMesh`——顶点
(V, 3)（m）、三角形 (T, 3) int64（指向顶点表）、顶点法向 (V, 3)（单位化）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MT_TRIANGLES",
    "TriangleMesh",
    "marching_tetrahedra",
]

# 立方体 8 角点的 (dx, dy, dz) 位偏移：角点编号 c = dx + 2·dy + 4·dz。
_CUBE_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=np.int64)

# 每立方体 6 个四面体：全部包含体对角线 (0, 7)，其余两个顶点取
# 六边形环 1 -> 3 -> 2 -> 6 -> 4 -> 5 -> 1 的 6 条棱（标准拆分）。
_TETS = np.array([
    [0, 7, 1, 3], [0, 7, 3, 2], [0, 7, 2, 6],
    [0, 7, 6, 4], [0, 7, 4, 5], [0, 7, 5, 1],
], dtype=np.int64)

# 四面体的 6 条棱：e_m 连接角点序 (c_i, c_j)，i < j（与 16 case 表配套）。
_TET_EDGES = np.array([
    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
], dtype=np.int64)

#: 16 case 表：case = Σ_k 2^k·[f(c_k) < 0]（角点 c_k 按 _TETS 行序），
#: 值为三角形列表，每个三角形是 3 个棱编号 (e0, e1, e2)。
#: 规则：0/4 个内点 -> 空表；1/3 个内点 -> 1 个三角形（内点/外点的 3 条邻棱）；
#: 2 个内点 -> 四边形拆 2 个三角形（拆分对角线在四面体内部，不跨面，
#: 故邻接四面体之间天然无拓扑歧义——MT 相对 MC 的教学优势，教程 5.1 ③(b)）。
MT_TRIANGLES: dict[int, tuple[tuple[int, int, int], ...]] = {
    0: (),
    1: ((0, 1, 2),),                    # 仅 c0 内：e0(0,1) e1(0,2) e2(0,3)
    2: ((0, 3, 4),),                    # 仅 c1 内：e0(0,1) e3(1,2) e4(1,3)
    3: ((1, 2, 4), (1, 4, 3)),          # c0,c1 内：四边形 e1 e2 e4 e3
    4: ((1, 3, 5),),                    # 仅 c2 内：e1(0,2) e3(1,2) e5(2,3)
    5: ((0, 2, 5), (0, 5, 3)),          # c0,c2 内：四边形 e0 e2 e5 e3
    6: ((0, 4, 5), (0, 5, 1)),          # c1,c2 内：四边形 e0 e4 e5 e1
    7: ((2, 4, 5),),                    # 仅 c3 外（=仅 c3 内的补）：e2 e4 e5
    8: ((2, 4, 5),),                    # 仅 c3 内：e2(0,3) e4(1,3) e5(2,3)
    9: ((0, 1, 5), (0, 5, 4)),          # c0,c3 内：四边形 e0 e1 e5 e4
    10: ((0, 2, 5), (0, 5, 3)),         # c1,c3 内（=case5 的补）
    11: ((1, 3, 5),),                   # 仅 c2 外（=case4 的补，变号棱同为 e1 e3 e5）
    12: ((1, 2, 4), (1, 4, 3)),         # c2,c3 内（=case3 的补）
    13: ((0, 3, 4),),                   # 仅 c1 外（=case2 的补，变号棱同为 e0 e3 e4）
    14: ((0, 1, 2),),                   # 仅 c0 外（=case1 的补）
    15: (),
}


@dataclass
class TriangleMesh:
    """三角网格（顶点表 + 面片表 + 顶点法向，教程 (1.8) 的网格表示）。

    Attributes:
        vertices: (V, 3) 顶点坐标，单位 m。
        triangles: (T, 3) 三角形顶点索引（int64，指向 ``vertices`` 行），
            取向统一为外法向（逆时针 = 从表面外侧看）。
        vertex_normals: (V, 3) 顶点法向（单位向量），由场梯度方向给出
            （教程 (5.2)），外正内负约定下指向物体外部。
    """

    vertices: np.ndarray
    triangles: np.ndarray
    vertex_normals: np.ndarray


def marching_tetrahedra(
    field: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> TriangleMesh:
    """在均匀体素标量场上提取零等值面（Marching Tetrahedra）。

    与主流的 Marching Cubes（Lorensen & Cline, SIGGRAPH 1987）相比，MT 把
    每个立方体先拆成 6 个四面体再查表：四面体只有 2^4 = 16 种符号配置、
    每种配置的三角化唯一（无 MC 15/33 类模板的二义面问题，教程 5.2 剖析的
    "结构性孔洞风险"被消除），代价是三角形数量增多——教学实现选 MT 是
    **代码量与正确性的权衡**：16 行手写表 + 30 行核心逻辑即可保证水密，
    MC 的 256 配置查表实现则远超教学篇幅（MC 的约定与歧义分析见教程
    第 05 章 5.1–5.2 节）。

    算法（对应教程 5.1 的流程，把"立方体"换成"四面体"）：
    1. 逐立方体拆 6 四面体，读 4 角点场值定 case（16 选 1，查 ``MT_TRIANGLES``）；
    2. 变号棱上求交点：t = f_a/(f_a - f_b)，p = p_a + t·(p_b - p_a)（教程 (5.1)）；
    3. 顶点去重：同一全局棱（两个格点号决定）只生成一个顶点——这是水密性的
       关键（相邻四面体共享棱上的交点数值完全一致）；
    4. 顶点法向 = 场梯度方向（教程 (5.2)，梯度用中心差分 + 棱上插值）；
    5. 三角形取向统一为"法向与场梯度同侧"（外正内负下即外法向，(1.3)/(1.13)）。

    Args:
        field: (n, n, n) 标量场采样，样本 (i, j, k) 位于
            ``origin + (i, j, k)·voxel_size``（零等值面即表面，回引 (1.2)）。
            允许 NaN：任一角点为 NaN（未观测，见
            ``tsdf.TSDFVolume.extraction_field``）的四面体整体跳过，
            用于权重掩码提取、避免未知区的幽灵内壁。
        origin: (3,) 场样本 (0, 0, 0) 的世界坐标，单位 m。
        voxel_size: 采样间距（体素边长），单位 m，取值 > 0。

    Returns:
        mesh: :class:`TriangleMesh`。场恒正/恒负（无变号棱）时返回空网格
        （顶点 (0, 3)、三角形 (0, 3)、法向 (0, 3)）。
    """
    field = np.asarray(field, dtype=float)
    if field.ndim != 3 or min(field.shape) < 2:
        raise ValueError(f"field 须为 (n, n, n)、n >= 2，当前 shape {field.shape}")
    n = field.shape[0]
    origin = np.asarray(origin, dtype=float)
    if not (field.shape[0] == field.shape[1] == field.shape[2]):
        raise ValueError("教学实现仅支持立方网格（三轴等长）")
    voxel = float(voxel_size)

    # ---------- 1. 立方体角点的全局量（格点号 / 坐标 / 场值 / 梯度） ----------
    n_cells = n - 1                                   # 每轴立方体数
    ci, cj, ck = np.meshgrid(np.arange(n_cells), np.arange(n_cells),
                             np.arange(n_cells), indexing="ij")
    cell_idx = ((ck.ravel() * n_cells) + cj.ravel()) * n_cells + ci.ravel()  # (C,)
    ii = cell_idx % n_cells
    jj = (cell_idx // n_cells) % n_cells
    kk = cell_idx // (n_cells * n_cells)
    n_cells_total = len(cell_idx)

    # 格点号 = ((k·n) + j)·n + i；角点 = 体角 + 位偏移。
    node_ids = np.stack([
        ((kk + dz) * n + (jj + dy)) * n + (ii + dx)
        for dx, dy, dz in _CUBE_CORNERS
    ], axis=0)                                        # (8, C)
    corner_pos = np.stack([
        origin + np.array([ii + dx, jj + dy, kk + dz]).T * voxel
        for dx, dy, dz in _CUBE_CORNERS
    ], axis=0)                                        # (8, C, 3)
    corner_val = field[
        node_ids // (n * n), (node_ids // n) % n, node_ids % n
    ]                                                 # (8, C)

    # 顶点法向 (5.2) 用场梯度：格点上中心差分，间距 = voxel（np.gradient 内部即该式）。
    # 注意轴序：np.gradient 沿数组轴返回 (d/daxis0, d/daxis1, d/daxis2) = (d/dz, d/dy, d/dx)，
    # 而坐标/法向矢量约定为 (x, y, z)，故翻转轴序对齐。
    grad_zyx = np.stack(np.gradient(field, voxel, voxel, voxel), axis=-1)  # (n, n, n, 3)
    grad = grad_zyx[..., ::-1]                                        # -> (dF/dx, dF/dy, dF/dz)
    corner_grad = grad[
        node_ids // (n * n), (node_ids // n) % n, node_ids % n
    ]                                                 # (8, C, 3)

    # ---------- 2. 逐四面体：case 表 + 变号棱求交 ----------
    total_nodes = n * n * n
    all_keys: list[np.ndarray] = []                   # 每三角形 3 个"全局棱键"
    all_pts: list[np.ndarray] = []                    # 对应交点坐标
    all_nrm: list[np.ndarray] = []                    # 对应插值法向
    for tet in _TETS:                                 # (4,) 四面体的 4 个立方体角点
        ids = node_ids[tet]                           # (4, C) 角点格点号
        vals = corner_val[tet]                        # (4, C)
        pos = corner_pos[tet]                         # (4, C, 3)
        grd = corner_grad[tet]                        # (4, C, 3)
        # 权重掩码提取：任一角点非有限（NaN = 未观测）的四面体整体跳过，
        # 避免未知区（+1 初值）与观测带边界之间生成幽灵内壁（见
        # tsdf.TSDFVolume.extraction_field 的说明）。
        usable = np.all(np.isfinite(vals), axis=0)    # (C,)
        if not usable.any():
            continue
        inside = (vals < 0.0) & usable[None, :]       # (4, C) 外正内负（(1.13)）
        case = (inside[0].astype(np.int64) | (inside[1].astype(np.int64) << 1)
                | (inside[2].astype(np.int64) << 2) | (inside[3].astype(np.int64) << 3))

        # 6 条棱的交点与键（键 = 有序格点对 -> 全局唯一，跨立方体一致）。
        edge_keys = np.zeros((6, n_cells_total), dtype=np.int64)
        edge_pts = np.zeros((6, n_cells_total, 3), dtype=float)
        edge_nrms = np.zeros((6, n_cells_total, 3), dtype=float)
        for e, (a, b) in enumerate(_TET_EDGES):
            fa, fb = vals[a], vals[b]
            crossed = inside[a] != inside[b]          # 严格变号（含 0 值退化时不算）
            if not crossed.any():
                continue
            # 教程 (5.1)：t = f_a / (f_a - f_b) ∈ (0, 1)；分子分母同号保证。
            tt = fa[crossed] / (fa[crossed] - fb[crossed])
            p = pos[a][crossed] + tt[:, None] * (pos[b][crossed] - pos[a][crossed])
            g = grd[a][crossed] + tt[:, None] * (grd[b][crossed] - grd[a][crossed])
            nrm = np.linalg.norm(g, axis=-1, keepdims=True)
            g = g / np.maximum(nrm, 1e-12)            # (5.2)：法向 = ∇F/‖∇F‖
            na, nb = ids[a][crossed], ids[b][crossed]
            lo = np.minimum(na, nb)
            hi = np.maximum(na, nb)
            edge_keys[e][crossed] = lo * total_nodes + hi   # 有序格点对 -> 唯一键
            edge_pts[e][crossed] = p
            edge_nrms[e][crossed] = g

        # 查 16 case 表，收集本四面体的三角形（棱键 + 交点 + 法向）。
        for c, tris in MT_TRIANGLES.items():
            if not tris:
                continue
            rows = np.nonzero(case == c)[0]
            if len(rows) == 0:
                continue
            cols = rows[None, :]                      # (1, m) 广播配 (3, 1)
            for tri in tris:                          # tri = (e0, e1, e2) 棱编号
                e = np.asarray(tri)[:, None]          # (3, 1)
                all_keys.append(edge_keys[e, cols].T)                    # (m, 3)
                all_pts.append(edge_pts[e, cols].transpose(1, 0, 2))     # (m, 3, 3)
                all_nrm.append(edge_nrms[e, cols].transpose(1, 0, 2))    # (m, 3, 3)

    if not all_keys:
        # 全场同号（空 SDF / 全负场）：等值面不存在，返回空网格。
        return TriangleMesh(
            vertices=np.zeros((0, 3)),
            triangles=np.zeros((0, 3), dtype=np.int64),
            vertex_normals=np.zeros((0, 3)),
        )

    # ---------- 3. 顶点去重（全局棱键）+ 组装 + 取向统一 ----------
    keys = np.concatenate(all_keys, axis=0)           # (M, 3) 每行 = 3 个顶点键
    pts = np.concatenate(all_pts, axis=0)             # (M, 3, 3)
    nrms = np.concatenate(all_nrm, axis=0)            # (M, 3, 3)
    flat_keys = keys.ravel()
    uniq, first_idx, inverse = np.unique(flat_keys, return_index=True, return_inverse=True)
    vertices = pts.reshape(-1, 3)[first_idx]          # 每个唯一棱只取首个交点
    vertex_normals = nrms.reshape(-1, 3)[first_idx]
    triangles = inverse.reshape(-1, 3).astype(np.int64)

    # 取向：三角形法向与"质心处场梯度"同向（外正内负下 = 外法向，(1.3)/(1.13)）。
    e1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    e2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    face_n = np.cross(e1, e2)
    grad_ref = vertex_normals[triangles].mean(axis=1)  # (T, 3) 质心梯度（单位化量的均值）
    flip = np.einsum("ij,ij->i", face_n, grad_ref) < 0.0
    swapped = triangles[flip][:, [0, 2, 1]]
    triangles[flip] = swapped

    return TriangleMesh(vertices=vertices, triangles=triangles,
                        vertex_normals=vertex_normals)
