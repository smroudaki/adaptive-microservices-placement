import networkx as nx
from microservices.type import MicroserviceNodeType
from yafs.application import Application, Message, fractional_selectivity
from yafs.topology import Topology


def dag_to_yafs_application(dag: nx.DiGraph) -> Application:
    """
    Creates YAFS Application objects based on networkx DAGs.

    Parameters:
    dag (nx.DiGraph): A directed acyclic graph representing the microservice architecture.

    Returns:
    Application: A YAFS Application object.
    """
    app = Application(name=dag.name)
    modules = []
    messages = {}

    # Handling messages
    for source, target, attrib in dag.edges(data=True):
        # Create a Message object for each edge in the DAG
        messages[attrib["name"]] = Message(
            attrib["name"],
            source,
            target,
            instructions=attrib["instructions"],
            bytes=attrib["bytes"],
        )

        # If the source node is a SOURCE type, add the message as a source message in the application
        if dag.nodes[source]["type"] == MicroserviceNodeType.SOURCE.value:
            app.add_source_messages(messages[attrib["name"]])

    for label, attrib in dag.nodes(data=True):
        # Handle source nodes separately
        if attrib["type"] == MicroserviceNodeType.SOURCE.value:
            modules.append({label: {"Type": attrib["type"]}})
        else:  # Handle module nodes
            modules.append(
                {
                    label: {
                        "Type": attrib["type"],
                        # "RAM": attrib["memory"],
                        # "storage": attrib["storage"],
                    }
                }
            )

        # Handle transmissions for module nodes
        if attrib["type"] == MicroserviceNodeType.MODULE.value:
            for *_, message_in in dag.in_edges(label, data="name"):
                for *_, message_out in dag.out_edges(label, data="name"):
                    app.add_service_module(
                        label,
                        messages[message_in],
                        messages[message_out],
                        fractional_selectivity,
                        threshold=1.0,
                    )
        # Handle sink nodes
        elif attrib["type"] == MicroserviceNodeType.SINK.value:
            for *_, message_in in dag.in_edges(label, data="name"):
                app.add_service_module(label, messages[message_in])

    # Set the modules in the application
    app.set_modules(modules)

    return app


def network_to_yafs_topology(graph: nx.Graph) -> Topology:
    """
    Creates a YAFS Topology using a networkx network graph.

    Parameters:
    graph (nx.Graph): A network graph representing the network topology.

    Returns:
    Topology: A YAFS Topology object.

    Raises:
    TypeError: If the input graph is not an instance of networkx.Graph.
    """
    if not isinstance(graph, nx.classes.graph.Graph):
        raise TypeError

    topology = Topology()
    topology.G = graph
    for label, attrib in graph.nodes(data=True):
        # Copy node attributes and set default uptime
        topology.nodeAttributes[label] = attrib.copy()
        topology.nodeAttributes[label]["uptime"] = (0, None)

    # Set the internal node ID counter
    topology.__idNode = len(graph.nodes)

    return topology
