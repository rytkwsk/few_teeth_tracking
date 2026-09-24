"""
few_teeth_tracking/extract_cluster_mesh.py

フル歯列 OBJ から「少数歯クラスタ」のサブメッシュを切り出すツール。
外部ライブラリ不要（numpy + 標準ライブラリのみ）。

切断方式について:
    以前はバウンディングボックス内に全頂点が収まる面だけを残す方式だったが、
    これは既存メッシュの三角形境界に沿って歯抜けに面を捨てるだけなので、
    切断線が階段状にガタつき、M3T のリージョンベーストラッキングで
    輪郭（シルエット）誤差の原因になっていた。
    現在は平面 bisect（幾何学的切断）方式を採用し、境界を通過する三角形を
    ちょうど平面上で分割することで、切断線を完全に直線的にしている。

使い方:
    python extract_cluster_mesh.py --input ../models/typodont.obj \
        --xmin -0.4 --xmax 0.4 \
        --ymin -0.5 --ymax 0.5 \
        --zmin -0.2 --zmax 0.5 \
        --output models/cluster.obj

バウンディングボックスの座標はモデル座標系（OBJ 内の生の数値）。
typodont.obj の MODEL_UNIT_IN_METER = 0.04 なら、
  1 model_unit = 4 cm に相当。

手順:
  1. test_tracker.py でフル歯列を追跡しながら視覚的に確認する。
  2. 切り出したい歯のおおよその座標範囲を --xmin/xmax 等で指定する。
  3. 生成された OBJ を few_teeth_tracking/models/ に置く。
  4. tracker.py の CLUSTER_MODEL_NAME を更新して起動する。
"""

import argparse
import os


def load_obj(path: str):
    """OBJ を読み込んで (vertices, faces, lines) を返す。
    faces: list of list of (v_idx, vt_idx_or_None, vn_idx_or_None)
    lines: 元のコメント行・mtllib 行など（非 v/vt/vn/f 行）
    """
    vertices = []  # list of [x, y, z]
    faces    = []  # list of list of (v0, v1, v2, ...) 0-indexed
    mtl_lines = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                ids = []
                for tok in line.split()[1:]:
                    try:
                        ids.append(int(tok.split("/")[0]) - 1)
                    except Exception:
                        pass
                if len(ids) >= 3:
                    faces.append(ids)
            else:
                mtl_lines.append(line)

    return vertices, faces, mtl_lines


def triangulate_faces(faces):
    """任意角数のポリゴンをファン三角形分割する（bisect の前処理）。"""
    tri_faces = []
    for f in faces:
        for k in range(1, len(f) - 1):
            tri_faces.append([f[0], f[k], f[k + 1]])
    return tri_faces


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def _clip_triangle(tri_coords, plane_co, plane_no, eps=1e-12):
    """三角形 (3 頂点座標) を平面で切断し、平面の内側 (d <= 0) だけを残す。

    戻り値は 0〜2 個の三角形（新しい座標のリスト）。
    d(v) = dot(v - plane_co, plane_no)
    """
    d = [sum((v[i] - plane_co[i]) * plane_no[i] for i in range(3)) for v in tri_coords]
    inside = [x <= eps for x in d]
    n_in = sum(inside)

    if n_in == 3:
        return [tri_coords]
    if n_in == 0:
        return []

    if n_in == 1:
        i_in = inside.index(True)
        i_o1 = (i_in + 1) % 3
        i_o2 = (i_in + 2) % 3
        v_in, v_o1, v_o2 = tri_coords[i_in], tri_coords[i_o1], tri_coords[i_o2]
        d_in, d_o1, d_o2 = d[i_in], d[i_o1], d[i_o2]
        p1 = _lerp(v_in, v_o1, d_in / (d_in - d_o1))
        p2 = _lerp(v_o2, v_in, d_o2 / (d_o2 - d_in))
        return [[v_in, p1, p2]]

    # n_in == 2
    i_out = inside.index(False)
    i_in1 = (i_out + 1) % 3
    i_in2 = (i_out + 2) % 3
    v_out, v_in1, v_in2 = tri_coords[i_out], tri_coords[i_in1], tri_coords[i_in2]
    d_out, d_in1, d_in2 = d[i_out], d[i_in1], d[i_in2]
    p_a = _lerp(v_out, v_in1, d_out / (d_out - d_in1))
    p_b = _lerp(v_in2, v_out, d_in2 / (d_in2 - d_out))
    return [[v_in1, v_in2, p_b], [v_in1, p_b, p_a]]


def bisect_mesh(vertices, faces, plane_co, plane_no):
    """三角形メッシュを平面で切断し、内側 (d <= 0) だけを残した新メッシュを返す。

    切断線上に新しい頂点をちょうど生成するため、既存の面境界に引きずられた
    ガタつきが出ない（bmesh.ops.bisect_plane と同等の処理）。
    """
    new_vertices = list(vertices)
    new_faces = []
    cache = {}  # (v_idx, v_idx) -> new vertex index, 同じ交点の重複生成を避ける

    def vid_for_coord(key, coord):
        if key in cache:
            return cache[key]
        new_vertices.append(coord)
        idx = len(new_vertices) - 1
        cache[key] = idx
        return idx

    for f in faces:
        tri_coords = [vertices[vi] for vi in f]
        d = [sum((tri_coords[i][k] - plane_co[k]) * plane_no[k] for k in range(3)) for i in range(3)]
        inside = [x <= 1e-12 for x in d]
        n_in = sum(inside)

        if n_in == 3:
            new_faces.append(list(f))
            continue
        if n_in == 0:
            continue

        clipped = _clip_triangle(tri_coords, plane_co, plane_no)
        # 切断で生じた新頂点をグローバル頂点リストに登録（既存頂点はインデックス流用）
        coord_to_existing = {tuple(round(c, 9) for c in tri_coords[i]): f[i] for i in range(3)}
        for tri in clipped:
            ids = []
            for coord in tri:
                key = tuple(round(c, 9) for c in coord)
                if key in coord_to_existing:
                    ids.append(coord_to_existing[key])
                else:
                    ids.append(vid_for_coord(key, list(coord)))
            new_faces.append(ids)

    return new_vertices, new_faces


def bisect_box(vertices, faces, x_range=None, y_range=None, z_range=None):
    """6 平面 (バウンディングボックス) で順番に bisect し、内側だけを残す。"""
    planes = []
    if x_range:
        planes.append(((x_range[1], 0, 0), (1, 0, 0)))   # x <= xmax
        planes.append(((x_range[0], 0, 0), (-1, 0, 0)))  # x >= xmin
    if y_range:
        planes.append(((0, y_range[1], 0), (0, 1, 0)))
        planes.append(((0, y_range[0], 0), (0, -1, 0)))
    if z_range:
        planes.append(((0, 0, z_range[1]), (0, 0, 1)))
        planes.append(((0, 0, z_range[0]), (0, 0, -1)))

    for plane_co, plane_no in planes:
        if not faces:
            break
        vertices, faces = bisect_mesh(vertices, faces, plane_co, plane_no)

    return vertices, faces


def keep_largest_connected_component(faces):
    """面のリストから、辺共有でつながった最大の連結成分だけを残す。

    元の typodont.obj には浮遊した微小なゴミポリゴン（元モデルのスキャンノイズ）が
    多数散らばっており、バウンディングボックスがたまたまそれらを拾ってしまうことがある。
    切り出したい歯本体は必ず1つの大きな連結成分になるはずなので、それ以外を捨てる。
    """
    edge_to_faces = {}
    for fi, f in enumerate(faces):
        n = len(f)
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            edge = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(edge, []).append(fi)

    adjacency = [[] for _ in faces]
    for face_ids in edge_to_faces.values():
        for i in face_ids:
            for j in face_ids:
                if i != j:
                    adjacency[i].append(j)

    visited = [False] * len(faces)
    components = []
    for start in range(len(faces)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            fi = stack.pop()
            comp.append(fi)
            for nb in adjacency[fi]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(comp)

    largest = max(components, key=len)
    if len(components) > 1:
        dropped = sum(len(c) for c in components if c is not largest)
        print(f"  浮遊断片を除去: {len(components) - 1} 個の孤立成分 ({dropped} 面) を破棄")

    return [faces[i] for i in largest]


def extract_cluster(
    vertices,
    faces,
    x_range=None,
    y_range=None,
    z_range=None,
):
    """バウンディングボックス内の頂点を含む面を抽出する。

    x_range / y_range / z_range: (min, max) タプルまたは None（制限なし）

    平面 bisect 方式: バウンディングボックスの各面ちょうどの位置で三角形を
    切断し、内側だけを残す。既存の面境界に頼らないため切断線が直線になる。
    """
    # bisect は計算コストがかかるので、先に浮遊ノイズ片を除去してから行う
    faces = keep_largest_connected_component(triangulate_faces(faces))

    new_vertices, selected_faces = bisect_box(vertices, faces, x_range, y_range, z_range)

    if not selected_faces:
        raise ValueError(
            "指定したバウンディングボックス内に面が見つかりませんでした。\n"
            "xmin/xmax 等の範囲を広げてみてください。"
        )

    # bisect で分離した孤立断片（曲面の折り返し等）が出ることがあるので再度除去
    selected_faces = keep_largest_connected_component(selected_faces)

    # 使用頂点の再インデックス
    used_vids = sorted({vi for f in selected_faces for vi in f})
    remap = {old: new for new, old in enumerate(used_vids)}

    final_vertices = [new_vertices[i] for i in used_vids]
    final_faces    = [[remap[vi] for vi in f] for f in selected_faces]

    return final_vertices, final_faces


def save_obj(path: str, vertices, faces, header_lines=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Cluster submesh extracted by extract_cluster_mesh.py\n")
        if header_lines:
            for line in header_lines:
                if line.startswith("#") or line.strip() == "":
                    f.write(line + "\n")
        f.write(f"# vertices: {len(vertices)}, faces: {len(faces)}\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in faces:
            toks = " ".join(str(vi + 1) for vi in face)
            f.write(f"f {toks}\n")
    print(f"保存: {path}  ({len(vertices)} 頂点, {len(faces)} 面)")


def main():
    parser = argparse.ArgumentParser(
        description="フル歯列 OBJ から少数歯クラスタのサブメッシュを切り出す"
    )
    parser.add_argument("--input",  required=True, help="入力 OBJ ファイルパス")
    parser.add_argument("--output", required=True, help="出力 OBJ ファイルパス")
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--zmin", type=float, default=None)
    parser.add_argument("--zmax", type=float, default=None)
    parser.add_argument(
        "--info", action="store_true",
        help="モデルの頂点座標範囲を表示して終了（バウンディングボックス確認用）"
    )
    args = parser.parse_args()

    print(f"読み込み: {args.input}")
    vertices, faces, header_lines = load_obj(args.input)
    print(f"  {len(vertices)} 頂点, {len(faces)} 面")

    verts_arr = [[v[0], v[1], v[2]] for v in vertices]
    xs = [v[0] for v in verts_arr]
    ys = [v[1] for v in verts_arr]
    zs = [v[2] for v in verts_arr]
    print(f"  X: {min(xs):.3f} ~ {max(xs):.3f}")
    print(f"  Y: {min(ys):.3f} ~ {max(ys):.3f}")
    print(f"  Z: {min(zs):.3f} ~ {max(zs):.3f}")

    if args.info:
        return

    x_range = (args.xmin, args.xmax) if args.xmin is not None and args.xmax is not None else None
    y_range = (args.ymin, args.ymax) if args.ymin is not None and args.ymax is not None else None
    z_range = (args.zmin, args.zmax) if args.zmin is not None and args.zmax is not None else None

    if x_range is None and y_range is None and z_range is None:
        print("警告: バウンディングボックスが指定されていません。全頂点をコピーします。")
        new_verts, new_faces = vertices, faces
    else:
        new_verts, new_faces = extract_cluster(vertices, faces, x_range, y_range, z_range)

    save_obj(args.output, new_verts, new_faces, header_lines)

    # 削減率を表示
    v_ratio = len(new_verts) / len(vertices) * 100
    f_ratio = len(new_faces) / len(faces) * 100
    print(f"  削減率: 頂点 {v_ratio:.1f}%, 面 {f_ratio:.1f}%")
    print()
    print("次のステップ:")
    print("  1. 生成した OBJ を few_teeth_tracking/models/ に配置")
    print("  2. tracker.py の CLUSTER_MODEL_NAME を更新")
    print("  3. python tracker.py を起動 (初回は .meta 生成で数分かかります)")


if __name__ == "__main__":
    main()
