from typing import Dict
import numpy as np

from typing import List

# construct graph


class Vertex:
    def __init__(self, node_id: int) -> None:
        self.node_id = node_id

        # index -> edge weight mapping
        self.incoming = dict()
        self.outgoing = dict()

    def __str__(self) -> str:
        return str(self.__dict__)


class Graph:
    def __init__(self) -> None:
        self.nodes: List[Vertex] = list()

    def __str__(self) -> str:
        return str(self.nodes)


def find_incoming(node_id: int, matrix: np.ndarray) -> Dict:
    inc = dict()
    incoming = matrix[:, node_id]
    for i in range(incoming.shape[0]):
        if incoming[i] != -np.Inf:
            inc[i] = incoming[i]

    return inc


def find_outgoing(node_id: int, matrix: np.ndarray) -> Dict:
    out = dict()
    outgoing = matrix[node_id, :]
    for i in range(matrix.shape[0]):
        if outgoing[i] != -np.Inf:
            out[i] = outgoing[i]

    return out


def construct_graph(matrix: np.ndarray) -> Graph:
    graph = Graph()

    # square matrix, so shape index doesn't matter
    for node_id in range(matrix.shape[0]):
        v = Vertex(node_id)
        v.incoming = find_incoming(node_id, matrix)
        v.outgoing = find_outgoing(node_id, matrix)

        graph.nodes.append(v)

    return graph


def has_cycle(node1: Vertex, node2: Vertex) -> bool:
    if node1.node_id in node2.incoming.keys() and node2.node_id in node1.incoming.keys():
        return True
    else:
        return False
