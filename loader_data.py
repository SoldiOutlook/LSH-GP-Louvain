"""
loader_data.py
解析 a.owl 和 b.owl，提取所有 owl:Class 作为节点，构建无向图（仅有节点，无边）。
每个节点包含用于 LSH 的文本实例列表。
"""

import rdflib
from rdflib.namespace import RDF, RDFS, OWL
import networkx as nx
from typing import List, Dict, Optional
import re

# OBO 自定义属性命名空间
OBO_IN_OWL = rdflib.Namespace("http://www.geneontology.org/formats/oboInOwl#")

def normalize_text(text: str) -> str:
    """转小写，替换空格为下划线，保留字母数字和下划线"""
    text = text.lower().strip()
    # 替换空白字符为下划线
    text = re.sub(r'\s+', '_', text)
    # 移除非字母数字下划线字符（保留用于生物医学的连字符等可按需保留，这里先简化为只保留字母数字下划线）
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text

def extract_node_info(class_uri: rdflib.URIRef, graph: rdflib.Graph) -> Dict:
    """
    提取给定类的文本特征。
    返回字典: {
        'uri': 类 URI 字符串,
        'label': 原始标签,
        'synonyms': [syn1, syn2, ...],
        'definitions': [def1, ...],
        'ancestor_labels': [anc_label1, ...]
    }
    """
    info = {
        'uri': str(class_uri),
        'label': '',
        'synonyms': [],
        'definitions': [],
        'ancestor_labels': []
    }

    # rdfs:label
    for label in graph.objects(class_uri, RDFS.label):
        if isinstance(label, rdflib.Literal):
            info['label'] = str(label)
            break  # 取第一个

    # oboInOwl:hasRelatedSynonym
    for syn in graph.objects(class_uri, OBO_IN_OWL.hasRelatedSynonym):
        if isinstance(syn, rdflib.Literal):
            info['synonyms'].append(str(syn))
        elif isinstance(syn, rdflib.URIRef) or isinstance(syn, rdflib.BNode):
            # 有时同义词指向一个匿名资源，其 rdfs:label 存值
            syn_label = graph.value(syn, RDFS.label)
            if syn_label and isinstance(syn_label, rdflib.Literal):
                info['synonyms'].append(str(syn_label))

    # oboInOwl:hasDefinition
    for def_node in graph.objects(class_uri, OBO_IN_OWL.hasDefinition):
        if isinstance(def_node, rdflib.Literal):
            info['definitions'].append(str(def_node))
        elif isinstance(def_node, (rdflib.URIRef, rdflib.BNode)):
            def_label = graph.value(def_node, RDFS.label)
            if def_label and isinstance(def_label, rdflib.Literal):
                info['definitions'].append(str(def_label))

    return info

def get_ancestor_labels(class_uri: rdflib.URIRef, graph: rdflib.Graph) -> List[str]:
    """
    BFS 向上遍历 rdfs:subClassOf，收集所有祖先的 rdfs:label。
    """
    ancestors = set()
    queue = [class_uri]
    while queue:
        current = queue.pop(0)
        for parent in graph.objects(current, RDFS.subClassOf):
            if isinstance(parent, rdflib.URIRef) and parent not in ancestors:
                ancestors.add(parent)
                queue.append(parent)
    labels = []
    for anc in ancestors:
        label = graph.value(anc, RDFS.label)
        if label and isinstance(label, rdflib.Literal):
            labels.append(str(label))
    return labels

def load_owl_files(a_path: str = "data/a.owl",
                   b_path: str = "data/b.owl") -> nx.Graph:
    """
    加载 a.owl 和 b.owl，提取所有 owl:Class，构建无向图 G。
    节点属性 'texts' 为文本实例列表（每个为规范化的字符串）。
    返回 networkx.Graph。
    """
    G = nx.Graph()
    g = rdflib.Graph()

    print("=" * 60)
    print("加载 OWL 文件...")
    for path in [a_path, b_path]:
        try:
            g.parse(path, format="xml")
            print(f"  已加载: {path}")
        except Exception as e:
            print(f"  加载 {path} 失败: {e}")
            continue

    # 获取所有 owl:Class 实例
    classes = list(g.subjects(RDF.type, OWL.Class))
    print(f"找到 {len(classes)} 个 owl:Class")

    # 预处理：为所有类建立祖先标签索引（可优化为一次 BFS 遍历，此处简洁处理）
    print("提取祖先标签...")
    ancestor_cache = {}
    for c in classes:
        ancestor_cache[c] = get_ancestor_labels(c, g)

    print("构造节点...")
    node_count = 0
    for c in classes:
        info = extract_node_info(c, g)
        info['ancestor_labels'] = ancestor_cache.get(c, [])

        # 生成文本实例列表
        texts = []
        # 标签重复 3 次，强化权重
        if info['label']:
            nlabel = normalize_text(info['label'])
            texts.extend([nlabel] * 3)

        # 每个同义词单独一个实例
        for syn in info['synonyms']:
            texts.append(normalize_text(syn))

        # 每条定义单独一个实例
        for defn in info['definitions']:
            texts.append(normalize_text(defn))

        # 祖先标签拼接成一个实例，重复 2 次（结构信息）
        if info['ancestor_labels']:
            anc_text = ' '.join(normalize_text(a) for a in info['ancestor_labels'])
            texts.extend([anc_text] * 2)

        # 如果没有任何文本，使用 URI 片段作为后备
        if not texts:
            fallback = normalize_text(c.split('#')[-1] if '#' in str(c) else str(c))
            texts.append(fallback)

        # 添加节点
        node_id = str(c)  # 完整 URI 作为节点 ID
        G.add_node(node_id, uri=node_id, label=info['label'], texts=texts)
        node_count += 1

        if node_count % 500 == 0:
            print(f"  已处理 {node_count}/{len(classes)} 个类...")

    print(f"图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边（当前无连边）")
    print(f"节点示例: {list(G.nodes(data=True))[0]}")
    return G

if __name__ == "__main__":
    G = load_owl_files()
    print("\n测试完毕。")