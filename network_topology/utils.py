def find_parent(node, parent):
    """
    Path-compression optimization for union-find.

    Args:
        node: Node whose root we want.
        parent: Mapping of node => parent.

    Returns:
        The root of the node's set.
    """
    if parent[node] != node:
        parent[node] = find_parent(parent[node], parent)
    return parent[node]


def union_nodes(node1, node2, parent):
    """
    Union the sets to which node1 and node2 belong.

    Args:
        node1: First node.
        node2: Second node.
        parent: Mapping of node => parent.

    Returns:
        None.
    """
    root1 = find_parent(node1, parent)
    root2 = find_parent(node2, parent)
    if root1 != root2:
        parent[root2] = root1
