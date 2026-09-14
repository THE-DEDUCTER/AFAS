import networkx as nx

def get_accessible_route(start_node: str = "Room 101") -> str:
    """
    Calculates an accessible route to the nearest safe exit.
    This fulfills the (Stretch) Route guidance feature.
    """
    # Create a simple floor plan graph
    G = nx.Graph()
    
    # Add nodes (Rooms, Corridors, Exits)
    G.add_nodes_from(["Room 101", "Corridor A", "Corridor B", "Stairwell", "Main Exit", "Accessible Exit"])
    
    # Add edges with an accessibility flag
    G.add_edge("Room 101", "Corridor A", accessible=True)
    G.add_edge("Corridor A", "Stairwell", accessible=False) # Requires stairs
    G.add_edge("Stairwell", "Main Exit", accessible=False)
    G.add_edge("Corridor A", "Corridor B", accessible=True)
    G.add_edge("Corridor B", "Accessible Exit", accessible=True) # Ramp available
    
    try:
        # Create a subgraph filtering out non-accessible edges (like stairs)
        accessible_edges = [(u, v) for u, v, d in G.edges(data=True) if d['accessible']]
        G_acc = nx.Graph()
        G_acc.add_edges_from(accessible_edges)
        
        # Calculate shortest path to the safe accessible exit
        path = nx.shortest_path(G_acc, source=start_node, target="Accessible Exit")
        return " -> ".join(path)
    except nx.NetworkXNoPath:
        return "No accessible route found. Shelter in place and await rescue."
    except Exception as e:
        return "Guidance system offline."
