"""
evaluate.py
评估 LSH 阶段生成的初代无向图与真实匹配（reference.rdf）的准确率、召回率、F1。
"""

import os
import time
import xml.etree.ElementTree as ET
from typing import Set, Tuple
import networkx as nx

# ----------------------------- 解析 reference.rdf -----------------------------
def parse_reference_rdf(ref_path: str = "data/reference.rdf") -> Set[Tuple[str, str]]:
    """
    解析 Alignment 格式的 RDF，返回真实匹配对集合。
    每对为 (entity1_uri, entity2_uri)，顺序保持原样（mouse, human）。
    """
    tree = ET.parse(ref_path)
    root = tree.getroot()

    # 命名空间
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'align': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'
    }

    pairs = set()
    for map_elem in root.findall('.//align:map', ns):
        cell = map_elem.find('align:Cell', ns)
        if cell is None:
            continue
        e1 = cell.find('align:entity1', ns)
        e2 = cell.find('align:entity2', ns)
        if e1 is not None and e2 is not None:
            uri1 = e1.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
            uri2 = e2.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
            if uri1 and uri2:
                pairs.add((uri1, uri2))
    return pairs


# ----------------------------- 评估指标 -----------------------------
def evaluate_graph(graph: nx.Graph, reference_pairs: Set[Tuple[str, str]]) -> dict:
    """
    计算准确率、召回率、F1。
    graph: 无向图，节点 ID 为完整 URI。
    reference_pairs: 真实匹配对 (mouse_uri, human_uri)
    """
    # 图中现有的边集（无序对）
    graph_edges = set()
    for u, v in graph.edges():
        # 标准化为 (mouse_uri, human_uri) 顺序？ 参考对始终 mouse 在前 human 在后。
        # 我们需要判断图中有没有连接这两个节点，无论顺序。
        # 所以我们将图边转为无序的 frozenset 用于快速查找，同时也存储原始 tuple 以匹配参考对。
        graph_edges.add(frozenset((u, v)))

    # 统计真实匹配中哪些对应的节点存在于图中（可能有缺失）
    ref_in_graph = set()
    ref_missing_nodes = 0
    for e1, e2 in reference_pairs:
        if e1 in graph and e2 in graph:
            ref_in_graph.add(frozenset((e1, e2)))
        else:
            ref_missing_nodes += 1

    # TP: 预测边 ∩ 真实匹配（均在图中）
    tp = len(graph_edges & ref_in_graph)
    # FP: 预测边中不是真实匹配的
    fp = len(graph_edges) - tp
    # FN: 真实匹配中未被预测的（仅统计两个节点都在图中的）
    fn = len(ref_in_graph) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'total_predicted_edges': len(graph_edges),
        'total_ref_pairs': len(reference_pairs),
        'ref_pairs_both_in_graph': len(ref_in_graph),
        'ref_missing_nodes': ref_missing_nodes,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ----------------------------- 主程序 -----------------------------
if __name__ == "__main__":
    # 为了避免重新跑 LSH，优先加载已经保存的图
    graph_path = "data/initial_graph.gpickle"
    if os.path.exists(graph_path):
        print(f"从 {graph_path} 加载图...")
        G = nx.read_gpickle(graph_path)
        print(f"加载成功: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    else:
        print("未找到已保存的图，将重新运行 LSH 预处理（这可能需要几分钟）...")
        from loader_data import load_owl_files
        from LSH import apply_lsh_to_graph

        t0 = time.time()
        G = load_owl_files("data/a.owl", "data/b.owl")
        # 使用与您之前运行完全相同的参数
        G = apply_lsh_to_graph(G,
                               minhash_threshold=0.5,
                               num_perm=128,
                               simhash_K=10,
                               simhash_L=60,
                               max_features=8000)
        print(f"LSH 完成，耗时 {time.time()-t0:.1f}s")
        # 保存图供后续复用
        nx.write_gpickle(G, graph_path)
        print(f"图已保存至 {graph_path}")

    # 解析真实匹配
    ref_path = "data/reference.rdf"
    if not os.path.exists(ref_path):
        print(f"错误: 找不到参考文件 {ref_path}")
        exit(1)

    print("\n解析 reference.rdf ...")
    ref_pairs = parse_reference_rdf(ref_path)
    print(f"真实匹配对总数: {len(ref_pairs)}")

    # 评估
    metrics = evaluate_graph(G, ref_pairs)

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"真实匹配对数量: {metrics['total_ref_pairs']}")
    print(f"  - 其中两个实体都在图中的对数: {metrics['ref_pairs_both_in_graph']}")
    print(f"  - 缺失节点的参考对数（未纳入计算）: {metrics['ref_missing_nodes']}")
    print(f"预测边总数: {metrics['total_predicted_edges']}")
    print(f"TP (正确连边): {metrics['tp']}")
    print(f"FP (错误连边): {metrics['fp']}")
    print(f"FN (漏连边): {metrics['fn']}")
    print(f"准确率 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"召回率 (Recall):    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 分数:            {metrics['f1']:.4f}")
    print("=" * 60)