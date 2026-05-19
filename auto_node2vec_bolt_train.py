# coding=utf-8
"""
自动化 Node2Vec + Word2Vec 训练脚本。

功能：
1. 通过 Bolt 连接 TuGraph default 数据库。
2. 自动遍历图中有出边的节点。
3. 从这些节点出发执行 Node2Vec 二阶有偏随机游走。
4. 使用 Word2Vec/Skip-Gram 训练节点嵌入向量。
5. 保存随机游走结果和 embedding，方便写报告和截图。

运行前安装依赖：
    pip install neo4j gensim scipy

运行：
    python3 auto_node2vec_bolt_train.py
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from gensim.models import Word2Vec

load_dotenv()

# ==================== TuGraph Bolt 连接配置 ====================
URI = os.getenv("TUGRAPH_URI", "bolt://localhost:7687")
AUTH = (os.getenv("TUGRAPH_USER", "admin"), os.getenv("TUGRAPH_PASSWORD", ""))
DATABASE = os.getenv("TUGRAPH_DATABASE", "default")


# ==================== Node2Vec / Word2Vec 参数 ====================
WALK_LENGTH = 10       # 每条随机游走路径长度
NUM_WALKS = 5          # 每个起点重复游走次数
P = 1.0                # 返回参数，越小越倾向回到上一节点
Q = 1.0                # 进出参数，越小越倾向 DFS，越大越倾向 BFS
MAX_START_NODES = 20   # 最多选多少个有出边节点作为起点，避免输出过大
VECTOR_SIZE = 64       # embedding 维度
WINDOW = 5             # Word2Vec 上下文窗口
EPOCHS = 10            # 训练轮数
RANDOM_SEED = 42       # 固定随机种子，方便截图复现


# ==================== 输出文件 ====================
BASE_DIR = Path(__file__).resolve().parent
WALKS_JSON = BASE_DIR / "node2vec_walks.json"
EMBEDDINGS_JSON = BASE_DIR / "node2vec_embeddings.json"
SUMMARY_TXT = BASE_DIR / "node2vec_training_summary.txt"


random.seed(RANDOM_SEED)


def query_records(session, cypher, **params):
    """执行 Cypher 并返回 list[dict]。"""
    result = session.run(cypher, **params)
    return result.data()


def load_graph_from_tugraph(session, max_start_nodes):
    """
    从 TuGraph 读取图结构。

    返回：
    - adjacency: dict[int, list[int]]，出边邻接表
    - start_nodes: list[int]，有出边的内部顶点 ID
    - node_count: int，节点数
    - edge_count: int，边数
    """
    node_rows = query_records(session, "MATCH (n) RETURN count(n) AS node_count")
    node_count = node_rows[0]["node_count"] if node_rows else 0

    edge_rows_count = query_records(session, "MATCH ()-[r]->() RETURN count(r) AS edge_count")
    edge_count = edge_rows_count[0]["edge_count"] if edge_rows_count else 0

    # 查询所有有出边的节点，id(n) 是 TuGraph/Neo4j 风格的内部顶点 ID。
    start_rows = query_records(
        session,
        f"""
        MATCH (n)-[r]->(m)
        RETURN id(n) AS vid, count(r) AS out_degree
        ORDER BY out_degree DESC, vid ASC
        LIMIT {max_start_nodes}
        """,
    )
    start_nodes = [int(row["vid"]) for row in start_rows]

    # 一次性取出所有边，构造邻接表。Node2Vec 采样阶段不再反复访问数据库。
    edge_rows = query_records(
        session,
        """
        MATCH (n)-[r]->(m)
        RETURN id(n) AS src, id(m) AS dst
        """,
    )

    adjacency = {}
    for row in edge_rows:
        src = int(row["src"])
        dst = int(row["dst"])
        adjacency.setdefault(src, []).append(dst)
        adjacency.setdefault(dst, adjacency.get(dst, []))

    return adjacency, start_nodes, node_count, edge_count


def node2vec_walk(adjacency, start_node, walk_length, p, q):
    """
    执行一条 Node2Vec 二阶有偏随机游走。

    权重规则：
    - 候选节点等于上一节点：1 / p
    - 候选节点与上一节点相邻：1
    - 其他情况：1 / q
    """
    walk = [start_node]
    previous = None
    current = start_node

    for _ in range(walk_length - 1):
        neighbors = adjacency.get(current, [])
        if not neighbors:
            break

        # 第一步没有 previous，直接在当前节点邻居中均匀采样。
        if previous is None:
            next_node = random.choice(neighbors)
        else:
            previous_neighbors = set(adjacency.get(previous, []))
            weights = []
            for candidate in neighbors:
                if candidate == previous:
                    weights.append(1.0 / p)
                elif candidate in previous_neighbors:
                    weights.append(1.0)
                else:
                    weights.append(1.0 / q)
            next_node = random.choices(neighbors, weights=weights, k=1)[0]

        walk.append(next_node)
        previous = current
        current = next_node

    return walk


def generate_walks(adjacency, start_nodes, walk_length, num_walks, p, q):
    """从多个起点生成多条 Node2Vec 随机游走路径。"""
    walks = []
    for start_node in start_nodes:
        for _ in range(num_walks):
            walks.append(node2vec_walk(adjacency, start_node, walk_length, p, q))
    return walks


def train_embeddings(walks):
    """使用 Word2Vec Skip-Gram 训练节点 embedding。"""
    if not walks:
        raise RuntimeError("没有生成任何随机游走路径，无法训练 embedding。")

    sentences = [[str(node) for node in walk] for walk in walks]
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=1,
        sg=1,
        workers=1,
        epochs=EPOCHS,
        seed=RANDOM_SEED,
    )

    embeddings = {}
    for node in model.wv.index_to_key:
        embeddings[node] = [float(x) for x in model.wv[node].tolist()]
    return embeddings


def save_outputs(walks, embeddings, node_count, edge_count, start_nodes):
    """保存随机游走、embedding 和摘要。"""
    WALKS_JSON.write_text(
        json.dumps(walks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    EMBEDDINGS_JSON.write_text(
        json.dumps(embeddings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_lines = []
    preview_lines.append("Node2Vec 自动训练完成")
    preview_lines.append(f"数据库: {DATABASE}")
    preview_lines.append(f"节点数: {node_count}")
    preview_lines.append(f"边数: {edge_count}")
    preview_lines.append(f"选取的起点内部 vid: {start_nodes}")
    preview_lines.append(f"随机游走条数: {len(walks)}")
    preview_lines.append(f"embedding 节点数: {len(embeddings)}")
    preview_lines.append(f"embedding 维度: {VECTOR_SIZE}")
    preview_lines.append("")
    preview_lines.append("随机游走结果示例（前 5 条）:")
    for walk in walks[:5]:
        preview_lines.append(str(walk))
    preview_lines.append("")
    preview_lines.append("embedding 示例（前 3 个节点，每个只展示前 8 维）:")
    for node, vector in list(embeddings.items())[:3]:
        preview_lines.append(f"node={node}, embedding前8维={vector[:8]}")

    SUMMARY_TXT.write_text("\n".join(preview_lines) + "\n", encoding="utf-8")


def main():
    print("正在连接 TuGraph Bolt...")
    print(f"URI: {URI}")
    print(f"Database: {DATABASE}")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=DATABASE) as session:
            print("正在读取图结构并查找有出边的节点...")
            adjacency, start_nodes, node_count, edge_count = load_graph_from_tugraph(
                session,
                MAX_START_NODES,
            )

    print(f"节点数: {node_count}")
    print(f"边数: {edge_count}")
    print(f"有出边起点 vid: {start_nodes}")

    if not start_nodes:
        raise RuntimeError(
            "当前图中没有找到任何有出边的节点。请先导入边数据，或确认连接的是 default 数据库。"
        )

    print("正在生成 Node2Vec 随机游走路径...")
    walks = generate_walks(adjacency, start_nodes, WALK_LENGTH, NUM_WALKS, P, Q)
    print(f"随机游走条数: {len(walks)}")
    print("随机游走示例:")
    for walk in walks[:5]:
        print(walk)

    print("正在训练 Word2Vec/Skip-Gram embedding...")
    embeddings = train_embeddings(walks)
    print(f"embedding 节点数: {len(embeddings)}")
    for node, vector in list(embeddings.items())[:3]:
        print(f"node={node}, embedding前8维={vector[:8]}")

    save_outputs(walks, embeddings, node_count, edge_count, start_nodes)
    print("输出文件已保存：")
    print(f"- {WALKS_JSON}")
    print(f"- {EMBEDDINGS_JSON}")
    print(f"- {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
