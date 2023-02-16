from graph import Graph, has_cycle, construct_graph
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


def cle(graph: Graph) -> Graph:
    # ignore node 0, since ROOT
    for node in graph.nodes[1:]:
        # find max incoming node id
        incoming = node.incoming
        max_node_id = max(incoming, key=incoming.get)  # type: ignore
        max_node_weight = max(incoming.values())
        max_node = graph.nodes[max_node_id]

        # update
        graph.nodes[node.node_id].incoming = {max_node_id: max_node_weight}

        # check for cycle
        if (has_cycle(node, max_node)):
            # contract
            cycle = Graph()
            cycle.nodes = [node, max_node]

            contracted = contract_cycle(cycle, graph)
            for cycle_node in contracted.nodes:
                graph.nodes[cycle_node.node_id] = cycle_node

            # call cle recursively
            cle(graph)

    # mst found
    return graph


if __name__ == "__main__":

    adjacency_matrix = np.array([
        [-np.Inf, 9, 10, 9],
        [-np.Inf, -np.Inf, 20, 3],
        [-np.Inf, 30, -np.Inf, 30],
        [-np.Inf, 11, 0, -np.Inf]
    ])  # type: ignore
    graph = construct_graph(adjacency_matrix)
    graph = cle(graph)

    for node in graph.nodes:
        print(node.node_id, node.incoming)
