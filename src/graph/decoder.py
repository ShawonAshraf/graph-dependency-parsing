from .graph import Graph, has_cycle, construct_graph
import numpy as np


def contract_cycle(cycle: Graph, graph: Graph) -> Graph:
    cycle_node_ids = [node.node_id for node in cycle.nodes]
    # print("nodes in cyle ", cycle_node_ids)

    # for outgoing nodes from the cycle
    max_outgoing_weight = -np.Inf
    max_outgoing_id = -np.Inf
    max_outgoing_from = -np.Inf

    # for incoming nodes to the cycle
    max_incoming_id = -np.Inf
    max_incoming_weight = -np.Inf
    max_incoming_to = -np.Inf

    for i, c_node in enumerate(cycle.nodes):
        for j, g_node in enumerate(graph.nodes):
            if j in cycle_node_ids:
                continue

            # check for outgoing
            if c_node.node_id in g_node.incoming.keys():
                # we have a match
                if g_node.outgoing[c_node.node_id] > max_outgoing_weight:
                    max_outgoing_weight = g_node.outgoing[c_node.node_id]
                    max_outgoing_id = g_node.node_id
                    max_outgoing_from = c_node.node_id

            # incoming
            if c_node.node_id in g_node.outgoing.keys():
                t = g_node.outgoing[c_node.node_id] + \
                    cycle.nodes[(i + 1) % 2].incoming[c_node.node_id]
                if t > max_incoming_weight:
                    max_incoming_weight = t
                    max_incoming_id = g_node.node_id
                    max_incoming_to = c_node.node_id

    # update cycle // resolve
    for node in cycle.nodes:
        if node.node_id == max_incoming_to:
            node.incoming = {max_incoming_id: max_incoming_weight}
        elif node.node_id == max_outgoing_from:
            node.outgoing = {max_outgoing_id: max_outgoing_weight}

    # for node in cycle.nodes:
    #     print(node)

    # print("max out  from cycle", max_outgoing_id)
    # print("max in to cycle ", max_incoming_id)

    return cycle


# source (modified from) : https://github.com/tdozat/Parser-v2/blob/master/parser/misc/mst.py
def find_roots(edges):
    """Return a list of vertices that were considered root by a dependent."""
    return np.where(edges[1:] == 0)[0] + 1


def make_root(probs, root, eta=1e-9):
    """Make specified vertex (index) root and nothing else."""
    probs = np.array(probs)
    probs[1:, 0] = 0
    probs[root, :] = 0
    probs[root, 0] = 1
    probs /= np.sum(probs + eta, axis=1, keepdims=True)
    return probs


def score_edges(probs, edges, eta=1e-9):
    """score a graph (so we can choose the best one)"""
    return np.sum(np.log(probs[np.arange(1, len(probs)), edges[1:]] + eta))


def get_best_graph(probs):
    """
    Returns the best graph, applying the CLE algorithm and making sure
    there is only a single root.
    """

    # zero out the diagonal (no word can be its own head)
    probs *= 1 - np.eye(len(probs)).astype(np.float32)
    probs[0] = 0  # zero out first row (root points to nothing else)
    probs[0, 0] = 1  # root points to itself
    probs /= np.sum(probs, axis=1, keepdims=True)  # normalize

    # apply CLE algorithm
    # edges = chu_liu_edmonds(probs)
    edges = greedy(probs)

    # deal with multiple roots
    roots = find_roots(edges)
    best_edges = edges
    best_score = -np.inf
    if len(roots) > 1:
        # print("more than 1 root!", roots)
        for root in roots:
            # apply CLE again with each of the possible roots fixed as the root
            # we return the highest scoring graph
            probs_ = make_root(probs, root)
            # edges_ = chu_liu_edmonds(probs_)
            edges_ = greedy(probs_)
            score = score_edges(probs_, edges_)
            if score > best_score:
                best_edges = edges_
                best_score = score

    return best_edges


def find_cycles(edges):
    """
    Finds cycles in a graph. Returns empty list if no cycles exist.
    Cf. https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    """

    vertices = np.arange(len(edges))
    indices = np.zeros_like(vertices) - 1
    lowlinks = np.zeros_like(vertices) - 1
    stack = []
    onstack = np.zeros_like(vertices, dtype=np.bool_)
    current_index = 0
    cycles = []

    def _strong_connect(vertex, current_index):
        indices[vertex] = current_index
        lowlinks[vertex] = current_index
        stack.append(vertex)
        current_index += 1
        onstack[vertex] = True

        for vertex_ in np.where(edges == vertex)[0]:
            if indices[vertex_] == -1:
                current_index = _strong_connect(vertex_, current_index)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[vertex_])
            elif onstack[vertex_]:
                lowlinks[vertex] = min(lowlinks[vertex], indices[vertex_])

        if lowlinks[vertex] == indices[vertex]:
            cycle = []
            vertex_ = -1
            while vertex_ != vertex:
                vertex_ = stack.pop()
                onstack[vertex_] = False
                cycle.append(vertex_)
            if len(cycle) > 1:
                cycles.append(np.array(cycle))
        return current_index

    for vertex in vertices:
        if indices[vertex] == -1:
            current_index = _strong_connect(vertex, current_index)

    return cycles


def cle(probs) -> Graph:
    vertices = np.arange(len(probs))
    edges = np.argmax(probs, axis=1)
    cycles = find_cycles(edges)

    if cycles:
        # print("found cycle, fixing...")
        cycle_vertices = cycles.pop()  # (c)
        non_cycle_vertices = np.delete(vertices, cycle_vertices)  # (nc)
        cycle_edges = edges[cycle_vertices]  # (c)

        # get rid of cycle nodes
        non_cycle_probs = np.array(
            probs[non_cycle_vertices, :][:, non_cycle_vertices])  # (nc x nc)

        # add a node representing the cycle
        # (nc+1 x nc+1)
        non_cycle_probs = np.pad(non_cycle_probs, [[0, 1], [0, 1]], 'constant')

        # probabilities of heads outside the cycle
        # (c x nc) / (c x 1) = (c x nc)
        backoff_cycle_probs = probs[cycle_vertices][:, non_cycle_vertices] / \
                              probs[cycle_vertices, cycle_edges][:, None]

        # probability of a node inside the cycle depending on
        # something outside the cycle
        # max_0(c x nc) = (nc)
        non_cycle_probs[-1, :-1] = np.max(backoff_cycle_probs, axis=0)

        # probability of a node outside the cycle depending on
        # something inside the cycle
        # max_1(nc x c) = (nc)
        non_cycle_probs[:-1, -1] = np.max(
            probs[non_cycle_vertices][:, cycle_vertices], axis=1)

        # (nc+1)
        non_cycle_edges = cle(non_cycle_probs)

        # This is the best source vertex into the cycle
        non_cycle_root, non_cycle_edges = non_cycle_edges[-1], non_cycle_edges[:-1]  # in (nc)
        source_vertex = non_cycle_vertices[non_cycle_root]  # in (v)

        # This is the vertex in the cycle we want to change
        cycle_root = np.argmax(backoff_cycle_probs[:, non_cycle_root])  # in (c)
        target_vertex = cycle_vertices[cycle_root]  # in (v)
        edges[target_vertex] = source_vertex

        # update edges with any other changes
        mask = np.where(non_cycle_edges < len(non_cycle_probs) - 1)
        edges[non_cycle_vertices[mask]] = non_cycle_vertices[non_cycle_edges[mask]]
        mask = np.where(non_cycle_edges == len(non_cycle_probs) - 1)

        # FIX
        stuff = np.argmax(probs[non_cycle_vertices][:, cycle_vertices], axis=1)
        stuff2 = cycle_vertices[stuff]
        stuff3 = non_cycle_vertices[mask]
        edges[stuff3] = stuff2[mask]

    return edges


# run bot construct and decode in one function
def construct_and_decode(matrix: np.ndarray) -> Graph:
    g = construct_graph(matrix)
    return cle(g)
