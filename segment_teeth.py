"""
few_teeth_tracking/segment_teeth.py

フル歯列 OBJ から「歯の輪郭に沿った」少数歯クラスタを切り出すツール。
外部ライブラリ: numpy のみ。

extract_cluster_mesh.py (bisect / 平面切断) との違い:
    bisect 方式は境界がバウンディングボックスの直方体になるため、
    対象歯の隣にある歯を途中で寸断してしまう問題があった。
    本スクリプトは高さ（歯冠の突出方向）を基準にした
    ウォーターシェッド分割で、歯と歯の間の谷（本物の解剖学的境界）を
    自動検出し、その境界線上でのみ切断する。対象歯の歯冠は
    一切削られず、隣接歯との境界だけが自然な谷に沿って切れる。

手順:
  1. フル歯列から最大連結成分のみを残す（浮遊ノイズ除去）
  2. 頂点位置に反復ラプラシアン平滑化をかけ、局所ノイズを均した
     「大局的な高さ」を求める
  3. 対象領域内で局所的に最も高い点（各歯の歯冠頂点）をシードにして
     ウォーターシェッド分割（高い方から低い方へ領域を拡張）
  4. 指定した中心座標に最も近い N 個の歯を「対象」として選び、
     対象歯 + 境界稜線上の頂点を残りのメッシュから抽出

使い方:
    python segment_teeth.py --input ../models/typodont.obj \
        --output models/cluster.obj \
        --search-xmin -0.9 --search-xmax 0.9 \
        --search-zmin -1.4 --search-zmax -0.5 \
        --target-x 0.0 --target-z -1.0 --num-teeth 3
"""

import argparse
import pickle

import numpy as np

from extract_cluster_mesh import (
    load_obj, save_obj, triangulate_faces, keep_largest_connected_component, bisect_mesh,
)


def compute_concavity(V, F, Vs):
    """頂点ごとの凹凸度を求める。正 = 凹 (歯肉側), 負 = 凸 (歯冠側)。

    Vs (heavily smoothed positions) と元の頂点位置 V の差を、元メッシュの
    法線方向に投影する。歯冠は滑らかな凸面、歯根元〜歯肉は凹面になるため、
    この値で歯冠と歯肉のおおまかな境界（歯肉縁）を判別できる。
    """
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(a, b)
    vn = np.zeros_like(V)
    np.add.at(vn, F[:, 0], fn)
    np.add.at(vn, F[:, 1], fn)
    np.add.at(vn, F[:, 2], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vn = vn / norms
    diff = Vs - V
    return np.einsum('ij,ij->i', diff, vn)


def build_adjacency(vertex_count, faces):
    neighbors = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def laplacian_smooth(V, faces, iterations):
    """一様ラプラシアン平滑化。局所ノイズを均して大局的な形状を得る。"""
    Vs = V.copy()
    F = np.asarray(faces, dtype=np.int64)
    for _ in range(iterations):
        nsum = np.zeros_like(Vs)
        ncount = np.zeros(len(Vs))
        for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
            idx = F[:, i]
            np.add.at(nsum, idx, Vs[F[:, j]] + Vs[F[:, k]])
            np.add.at(ncount, idx, 2)
        ncount[ncount == 0] = 1
        Vs = nsum / ncount[:, None]
    return Vs


def find_crown_peaks(V, smoothed_height, neighbors, region_set, min_height):
    """領域内で、平滑化した高さが近傍のどの頂点よりも高い頂点（歯冠候補）を探す。"""
    peaks = []
    for vid in region_set:
        if smoothed_height[vid] < min_height:
            continue
        nb = [j for j in neighbors[vid] if j in region_set]
        if nb and all(smoothed_height[j] <= smoothed_height[vid] for j in nb):
            peaks.append(vid)
    return peaks


def watershed_segment(V, smoothed_height, neighbors, region_set, seed_labels, max_radius=None):
    """高い方から低い方へ領域を広げ、複数のラベルが接する頂点を境界 (-2) とする。

    max_radius を指定すると、各ラベルの領域はそのシード頂点からの
    XZ 平面上の距離（高さ Y 方向は無視）が max_radius を超えて
    広がらない。歯冠から歯肉までの Y 方向の伸びは正常な広がりだが、
    歯列表面には低い鞍部を経由してシードから水平方向に遠く離れた
    場所まで地続きになっている箇所があり、制限なしだと稜線を越えて
    全く別の歯の領域まで回り込んでしまうことがあるため。

    戻り値: {vid: label} (label >= 0 は所属ラベル, -2 は境界, -1 は未到達)
    """
    label = {vid: -1 for vid in region_set}
    seed_coord = {}
    for vid, lbl in seed_labels.items():
        if vid in label:
            label[vid] = lbl
            seed_coord[lbl] = V[vid]

    order = sorted(region_set, key=lambda i: -smoothed_height[i])
    for vid in order:
        if label[vid] != -1:
            continue
        nb_labels = set(
            label[j] for j in neighbors[vid] if j in region_set and label[j] >= 0
        )
        if max_radius is not None:
            vx, vz = V[vid, 0], V[vid, 2]
            nb_labels = {
                lbl for lbl in nb_labels
                if (vx - seed_coord[lbl][0]) ** 2 + (vz - seed_coord[lbl][2]) ** 2
                <= max_radius ** 2
            }
        if len(nb_labels) == 1:
            label[vid] = next(iter(nb_labels))
        elif len(nb_labels) > 1:
            label[vid] = -2

    # max_radius によって、周囲がすべてラベル付けされているのに自分だけ
    # 半径条件で弾かれ -1 (未到達) のまま残る孤立頂点が出ることがある。
    # そのままだと面の抜け（穴）になるため、実際のラベル付き近傍から
    # 半径を無視して穴埋めする。
    changed = True
    while changed:
        changed = False
        for vid in list(region_set):
            if label[vid] != -1:
                continue
            nb_labels = set(
                label[j] for j in neighbors[vid] if j in region_set and label[j] >= 0
            )
            if max_radius is not None:
                vx, vz = V[vid, 0], V[vid, 2]
                nb_labels = {
                    lbl for lbl in nb_labels
                    if (vx - seed_coord[lbl][0]) ** 2 + (vz - seed_coord[lbl][2]) ** 2
                    <= max_radius ** 2
                }
            if len(nb_labels) == 1:
                label[vid] = next(iter(nb_labels))
                changed = True
            elif len(nb_labels) > 1:
                label[vid] = -2
                changed = True
    return label


def segment_cluster(vertices, faces, search_box, target_xz, num_teeth,
                     smooth_iterations=25, min_crown_height=0.3, boundary_margin=0.08,
                     max_radius=0.4):
    """歯列メッシュから対象クラスタ (対象歯 N 本 + 境界) の面リストを返す。"""
    faces = keep_largest_connected_component(triangulate_faces(faces))
    V = np.array(vertices, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)

    neighbors = build_adjacency(len(V), F)
    used = np.array([len(neighbors[i]) > 0 for i in range(len(V))])

    xmin, xmax, zmin, zmax, ymin = search_box
    region_mask = used & (V[:, 0] >= xmin) & (V[:, 0] <= xmax) & \
        (V[:, 2] >= zmin) & (V[:, 2] <= zmax) & (V[:, 1] > ymin)
    region_set = set(np.where(region_mask)[0].tolist())
    if not region_set:
        raise ValueError("search-box 内に頂点が見つかりませんでした。範囲を広げてください。")

    Vs = laplacian_smooth(V, F, smooth_iterations)
    peaks = find_crown_peaks(V, Vs[:, 1], neighbors, region_set, min_crown_height)
    if len(peaks) < num_teeth:
        raise ValueError(
            f"歯冠候補が {len(peaks)} 個しか見つかりませんでした (num_teeth={num_teeth})。\n"
            "search-box を広げるか min_crown_height を下げてください。"
        )

    tx, tz = target_xz
    peaks_sorted = sorted(peaks, key=lambda vid: (V[vid, 0] - tx) ** 2 + (V[vid, 2] - tz) ** 2)
    seed_labels = {vid: i for i, vid in enumerate(peaks_sorted)}
    target_labels = set(range(num_teeth))  # 距離が近い順の先頭 num_teeth 個が対象歯

    label = watershed_segment(V, Vs[:, 1], neighbors, region_set, seed_labels, max_radius=max_radius)

    # 対象ラベルのみ (境界を含まない) の範囲をまず求め、境界頂点を取り込む際の
    # 安全マージンとして使う。マージンを大きく外れた境界頂点（別領域まで
    # 蛇行してつながっている遠い稜線）を切り捨て、意図しない断片の混入を防ぐ。
    strict_target = [vid for vid, lbl in label.items() if lbl in target_labels]
    if not strict_target:
        raise ValueError("対象歯のラベルが見つかりませんでした。target-x/z を見直してください。")
    txs = [V[v, 0] for v in strict_target]
    tys = [V[v, 1] for v in strict_target]
    tzs = [V[v, 2] for v in strict_target]
    box = (
        min(txs) - boundary_margin, max(txs) + boundary_margin,
        min(tys) - boundary_margin,
        min(tzs) - boundary_margin, max(tzs) + boundary_margin,
    )

    def in_margin(vid):
        x, y, z = V[vid]
        return box[0] <= x <= box[1] and y >= box[2] and box[3] <= z <= box[4]

    # 境界 (-2) 頂点は、対象ラベルに直接隣接するものだけでなく、そこから
    # 境界頂点同士でつながっている稜線全体を辿って取り込む。直接隣接のみだと
    # 稜線の途中（対象ラベルの頂点に触れていない箇所）でメッシュが途切れ、
    # 小さな穴があいてしまうため。
    boundary_keep = set()
    frontier = [
        vid for vid, lbl in label.items()
        if lbl == -2 and in_margin(vid)
        and any(label.get(j, -9) in target_labels for j in neighbors[vid])
    ]
    boundary_keep.update(frontier)
    stack = list(frontier)
    while stack:
        vid = stack.pop()
        for j in neighbors[vid]:
            if label.get(j) == -2 and j not in boundary_keep and in_margin(j):
                boundary_keep.add(j)
                stack.append(j)

    keep_v = set()
    for vid, lbl in label.items():
        if not in_margin(vid):
            continue
        if lbl in target_labels:
            keep_v.add(vid)
    keep_v.update(boundary_keep)

    keep_faces = [f for f in F.tolist() if f[0] in keep_v and f[1] in keep_v and f[2] in keep_v]
    if not keep_faces:
        raise ValueError("対象歯の抽出結果が空でした。target-x/target-z や num_teeth を見直してください。")

    keep_faces = keep_largest_connected_component(keep_faces)
    concavity = compute_concavity(V, F, Vs)
    return V.tolist(), keep_faces, concavity, label, seed_labels, Vs


def crown_only_faces(faces, concavity, threshold=0.0):
    """凹凸度が threshold 以下 (凸=歯冠側) の頂点だけで構成される面を残す。

    歯肉との境界 (歯肉縁) 付近で凹凸度がちょうど threshold をまたぐため、
    ここで切ることでおおよそ歯冠のみを取り出せる。歯根は含まれない
    （スキャンデータ自体に歯根形状がないため、歯冠のみの近似となる）。
    """
    return [f for f in faces if all(concavity[vi] <= threshold for vi in f)]


def crown_only_faces_by_height(V, faces, label, seed_labels, Vs, crown_depth=0.35):
    """各対象歯の頂点から、その歯のピーク高さより crown_depth 以上低い頂点を除いて
    歯冠のみを残す。歯ごとに独立した基準高さを使うため、per-vertex の凹凸度
    しきい値よりも境界がなめらかになる。

    seed_labels: {vid: label} 各対象歯のシード頂点
    label: watershed_segment が返す {vid: label} (>=0 が対象歯, -2 が境界)
    """
    peak_height = {lbl: Vs[vid, 1] for vid, lbl in seed_labels.items()}

    def keep(vid):
        lbl = label.get(vid, -9)
        if lbl not in peak_height:
            return False
        return V[vid, 1] >= peak_height[lbl] - crown_depth

    return [f for f in faces if all(keep(vi) for vi in f)]


def crown_only_mesh_bisect(V, faces, label, seed_labels, Vs, crown_depth=0.35, max_radius=None):
    """各対象歯ごとに、ピーク高さ - crown_depth の水平面で bisect (幾何学的切断)
    して歯冠のみを取り出す。

    per-vertex のしきい値判定 (crown_only_faces_by_height) だと、面ごとに
    「全頂点がしきい値以上」を満たすかどうかで判定するため、既存メッシュの
    面境界に沿ってガタガタに切れてしまう（extract_cluster_mesh.py の
    bisect 導入前の問題と同じ現象）。bisect ならしきい値の高さちょうどの
    位置に新しい頂点を生成して切るので、切断線が完全に滑らかになる。

    戻り値: (vertices, faces) の完全なメッシュ（歯ごとに独立した頂点番号に
    詰め直し済み、歯根が無いため歯ごとに非連結な島になるのが正常）。
    """
    peak_height = {lbl: Vs[vid, 1] for vid, lbl in seed_labels.items()}
    peak_xz = {lbl: (V[vid, 0], V[vid, 2]) for vid, lbl in seed_labels.items()}
    target_labels = set(peak_height.keys())

    combined_vertices = []
    combined_faces = []
    V_list = V.tolist()

    for lbl in sorted(target_labels):
        plane_y = peak_height[lbl] - crown_depth
        px, pz = peak_xz[lbl]
        # 境界 (-2) 頂点は「どこかで2つの領域が接した」印にすぎず、
        # そのシードからの距離に制限がない。そのまま含めると、この歯とは
        # 無関係な遠くの境界まで拾ってしまい、切断面ぎりぎりの高さで
        # 孤立した尖った破片（裏側のギザギザ）の原因になる。
        # そのためこの歯自身のシードから max_radius 以内の頂点だけに絞る。
        margin = 0.1
        sub_vids = set()
        for vid, l in label.items():
            if l != lbl and l != -2:
                continue
            if V[vid, 1] < plane_y - margin:
                continue
            if max_radius is not None:
                dx = V[vid, 0] - px
                dz = V[vid, 2] - pz
                if dx * dx + dz * dz > max_radius ** 2:
                    continue
            sub_vids.add(vid)
        sub_faces = [f for f in faces if all(v in sub_vids for v in f)]
        if not sub_faces:
            continue
        new_verts, clipped_faces = bisect_mesh(
            V_list, sub_faces, plane_co=(0, plane_y, 0), plane_no=(0, -1, 0)
        )
        clipped_faces = keep_largest_connected_component(clipped_faces)
        used = sorted({vi for f in clipped_faces for vi in f})
        remap = {old: new + len(combined_vertices) for new, old in enumerate(used)}
        combined_vertices.extend(new_verts[i] for i in used)
        combined_faces.extend([remap[vi] for vi in f] for f in clipped_faces)

    return combined_vertices, combined_faces


def main():
    parser = argparse.ArgumentParser(
        description="フル歯列 OBJ から歯の輪郭を保ったまま少数歯クラスタを切り出す"
                     "（ウォーターシェッド分割、歯間の谷でのみ切断）"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--search-xmin", type=float, required=True)
    parser.add_argument("--search-xmax", type=float, required=True)
    parser.add_argument("--search-zmin", type=float, required=True)
    parser.add_argument("--search-zmax", type=float, required=True)
    parser.add_argument("--search-ymin", type=float, default=0.1,
                         help="歯肉側の探索下限（このY未満は最初から除外）")
    parser.add_argument("--target-x", type=float, required=True,
                         help="残したい歯列の中心付近の X 座標")
    parser.add_argument("--target-z", type=float, required=True,
                         help="残したい歯列の中心付近の Z 座標")
    parser.add_argument("--num-teeth", type=int, default=3,
                         help="target-x/z に近い方から何本の歯を残すか")
    parser.add_argument("--smooth-iterations", type=int, default=25)
    parser.add_argument("--min-crown-height", type=float, default=0.3,
                         help="この Y 未満はシード候補にしない（歯肉側のノイズ除け）")
    parser.add_argument("--max-radius", type=float, default=0.4,
                         help="各歯のシードからの領域拡張をこの半径までに制限する"
                              "（歯列の端や広い歯肉に沿って領域が回り込むのを防ぐ）")
    parser.add_argument("--crown-only-output", default=None,
                         help="指定すると、凹凸度から歯肉を除いた歯冠のみの"
                              "OBJ も追加で書き出す（歯根は含まれない近似）")
    parser.add_argument("--crown-threshold", type=float, default=0.0,
                         help="凹凸度方式を使う場合の歯冠/歯肉のしきい値（--crown-method concavity 用）")
    parser.add_argument("--crown-depth", type=float, default=0.35,
                         help="各歯のピーク高さからこの深さまでを歯冠として残す"
                              "（--crown-method height、デフォルト）")
    parser.add_argument("--crown-method", choices=["bisect", "height", "concavity"], default="bisect",
                         help="歯冠/歯肉の分け方。bisect=各歯のピーク高さ-crown-depth の"
                              "水平面で幾何学的に切断（推奨、切断線が完全に滑らか）。"
                              "height/concavity は面単位のしきい値判定で切断線がガタつくため"
                              "比較・デバッグ用に残してあるだけで非推奨")
    args = parser.parse_args()

    print(f"読み込み: {args.input}")
    vertices, faces, _ = load_obj(args.input)
    print(f"  {len(vertices)} 頂点, {len(faces)} 面")

    search_box = (args.search_xmin, args.search_xmax,
                  args.search_zmin, args.search_zmax, args.search_ymin)
    new_vertices, new_faces, concavity, label, seed_labels, Vs = segment_cluster(
        vertices, faces, search_box,
        (args.target_x, args.target_z), args.num_teeth,
        smooth_iterations=args.smooth_iterations,
        min_crown_height=args.min_crown_height,
        max_radius=args.max_radius,
    )

    def remap_and_save(path, face_list):
        used_vids = sorted({vi for f in face_list for vi in f})
        remap = {old: new for new, old in enumerate(used_vids)}
        vs = [new_vertices[i] for i in used_vids]
        fs = [[remap[vi] for vi in f] for f in face_list]
        save_obj(path, vs, fs)

    remap_and_save(args.output, new_faces)

    if args.crown_only_output:
        # 歯冠のみにすると歯の本数だけ島に分かれるのが正常なので、
        # 単一の最大連結成分には絞らない（絞ると1本しか残らない）。
        Va = np.array(new_vertices, dtype=np.float64)
        if args.crown_method == "bisect":
            crown_vertices, crown_faces = crown_only_mesh_bisect(
                Va, new_faces, label, seed_labels, Vs, crown_depth=args.crown_depth,
                max_radius=args.max_radius,
            )
            if not crown_faces:
                raise ValueError(
                    "crown-depth で歯冠側の面が残りませんでした。値を大きくしてください。"
                )
            save_obj(args.crown_only_output, crown_vertices, crown_faces)
        else:
            if args.crown_method == "height":
                crown_faces = crown_only_faces_by_height(
                    Va, new_faces, label, seed_labels, Vs, crown_depth=args.crown_depth
                )
            else:
                crown_faces = crown_only_faces(new_faces, concavity, args.crown_threshold)
            if not crown_faces:
                raise ValueError(
                    "指定したしきい値で歯冠側の面が残りませんでした。値を見直してください。"
                )
            remap_and_save(args.crown_only_output, crown_faces)


if __name__ == "__main__":
    main()
