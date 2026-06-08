import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
from typing import List, Tuple, Set
import time
import itertools

def build_minhash_signatures(texts: List[str], num_perm: int = 128) -> List[MinHash]:
    signatures = []
    for text in texts:
        m = MinHash(num_perm=num_perm)
        for ng in [2, 3]:
            for i in range(len(text) - ng + 1):
                m.update(text[i:i+ng].encode('utf-8'))
        signatures.append(m)
    return signatures

def minhash_lsh(graph: nx.Graph, threshold: float = 0.5, num_perm: int = 128) -> Set[Tuple[str, str]]:
    t0 = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    instance_map = []

    nodes = list(graph.nodes(data=True))
    total_instances = 0
    for node_id, attr in nodes:
        texts = attr['texts']
        for idx, text in enumerate(texts):
            m = MinHash(num_perm=num_perm)
            for ng in [2, 3]:
                for i in range(len(text) - ng + 1):
                    m.update(text[i:i+ng].encode('utf-8'))
            lsh.insert(f"{node_id}::{idx}", m)
            instance_map.append((node_id, idx))
            total_instances += 1
    candidate_pairs = set()
    for node_id, attr in nodes:
        texts = attr['texts']
        for idx, text in enumerate(texts):
            m = MinHash(num_perm=num_perm)
            for ng in [2, 3]:
                for i in range(len(text) - ng + 1):
                    m.update(text[i:i+ng].encode('utf-8'))
            results = lsh.query(m)
            for res in results:
                other_node, _ = res.split("::")
                if other_node != node_id:
                    pair = tuple(sorted((node_id, other_node)))
                    candidate_pairs.add(pair)

    edges = set()
    edges.update(candidate_pairs)

    t1 = time.time()
    return edges

def simhash_lsh(vectors: np.ndarray, K: int = 10, L: int = 50) -> Set[Tuple[int, int]]:
    n, dim = vectors.shape
    random_planes = np.random.randn(dim, L * K)
    projections = vectors @ random_planes
    bits = (projections > 0).astype(int)

    edges_set = set()
    for l in range(L):
        start = l * K
        end = (l + 1) * K
        hash_codes = bits[:, start:end]
        bucket_keys = hash_codes.dot(1 << np.arange(K)[::-1]
        buckets = {}
        for i, bk in enumerate(bucket_keys):
            buckets.setdefault(bk, []).append(i)
        for bk, inst_ids in buckets.items():
            if len(inst_ids) > 1:
                for a, b in itertools.combinations(inst_ids, 2):
                    edges_set.add(tuple(sorted((a, b))))
    return edges_set

def simhash_lsh_graph(graph: nx.Graph, K: int = 10, L: int = 60,
                      max_features: int = 8000) -> Set[Tuple[str, str]]:
    t0 = time.time()
    nodes = list(graph.nodes(data=True))
    instance_texts = []
    instance_node_map = []
    for node_id, attr in nodes:
        for text in attr['texts']:
            instance_texts.append(text)
            instance_node_map.append(node_id)
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4), max_features=max_features,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(instance_texts).toarray().astype(np.float32)
    print(f"特征维度: {X.shape[1]}")

    X = X - X.mean(axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    print(f"运行 SimHash LSH (K={K}, L={L})...")
    candidate_pairs_idx = simhash_lsh(X, K=K, L=L)

    node_pairs = set()
    for i, j in candidate_pairs_idx:
        n1 = instance_node_map[i]
        n2 = instance_node_map[j]
        if n1 != n2:
            node_pairs.add(tuple(sorted((n1, n2))))

    t1 = time.time()
    print(f"SimHash 候选节点对数量: {len(node_pairs)}, 耗时: {t1-t0:.2f}s")
    return node_pairs

def apply_lsh_to_graph(G: nx.Graph,
                       minhash_threshold: float = 0.5,
                       num_perm: int = 128,
                       simhash_K: int = 10,
                       simhash_L: int = 60,
                       max_features: int = 8000) -> nx.Graph:
    edges_minhash = minhash_lsh(G, threshold=minhash_threshold, num_perm=num_perm)
    edges_simhash = simhash_lsh_graph(G, K=simhash_K, L=simhash_L, max_features=max_features)
    all_edges = edges_minhash.union(edges_simhash)
    G.add_edges_from(all_edges)
    print(f"最终图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    if G.number_of_nodes() > 0:
        density = 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
        print(f"图密度: {density:.6f}")

    return G

if __name__ == "__main__":
    from loader_data import load_owl_files
    G = load_owl_files("data/a.owl", "data/b.owl")
    G = apply_lsh_to_graph(G,
                           minhash_threshold=0.5,
                           num_perm=128,
                           simhash_K=10,
                           simhash_L=60,
                           max_features=8000)
    print("\nLSH 阶段一完成。")