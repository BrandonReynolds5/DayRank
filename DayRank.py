import numpy as np

def day_rank(adjacency_matrix, personalized_vector, teleportation_probability=0.15, max_iterations=100, tol=1e-6):
    """
    Compute the DayRank score for each node considering a personalized vector.

    Parameters:
    - adjacency_matrix (np.array): Weighted adjacency matrix where entry (i, j) represents
                                   the weight of the link from node j to node i.
    - personalized_vector (np.array): The personalized teleportation vector reflecting user preferences.
    - teleportation_probability (float): The probability of randomly teleporting to any node.
    - max_iterations (int): The maximum number of iterations for convergence.
    - tol (float): Tolerance for checking convergence.

    Returns:
    - day_rank_scores (np.array): The DayRank scores for each node.
    """
    num_nodes = adjacency_matrix.shape[0]
    
    # Normalize the adjacency matrix to ensure columns sum to 1
    column_sums = adjacency_matrix.sum(axis=0)
    normalized_matrix = adjacency_matrix / column_sums
    
    # Initialize the DayRank scores with a uniform distribution
    day_rank_scores = np.ones(num_nodes) / num_nodes
    
    # Iteratively update the DayRank scores
    for iteration in range(max_iterations):
        # Perform the matrix-vector multiplication
        new_day_rank_scores = normalized_matrix.dot(day_rank_scores)
        
        # Add the teleportation probability
        new_day_rank_scores = teleportation_probability * personalized_vector + (1 - teleportation_probability) * new_day_rank_scores
        
        # Check for convergence
        if np.linalg.norm(new_day_rank_scores - day_rank_scores, 1) < tol:
            break
        
        day_rank_scores = new_day_rank_scores
    
    return day_rank_scores

def find_best_route(adjacency_matrix, day_rank_scores, start_node, daily_goals):
    """
    Determine the best route based on DayRank scores and the daily goals.

    Parameters:
    - adjacency_matrix (np.array): Weighted adjacency matrix.
    - day_rank_scores (np.array): The DayRank scores for each node.
    - start_node (int): The starting node (location).
    - daily_goals (list of int): The nodes representing the daily goals.

    Returns:
    - route (list of int): The sequence of nodes representing the best route.
    - total_distance (float): The total distance of the selected route in miles.
    - route_distances (list of float): The distances for each leg of the route in miles.
    """
    num_nodes = adjacency_matrix.shape[0]
    route = [start_node]
    current_node = start_node
    visited = set(route)
    total_distance = 0
    route_distances = []
    
    while daily_goals:
        # Select the next best node based on DayRank scores and adjacency matrix weights
        next_node = None
        max_score = -1
        for node in range(num_nodes):
            if node not in visited and adjacency_matrix[current_node, node] > 0:
                # Consider only nodes not yet visited and reachable from the current node
                score = day_rank_scores[node] * adjacency_matrix[current_node, node]
                if score > max_score:
                    max_score = score
                    next_node = node
        
        if next_node is None:
            # No further reachable nodes
            break
        
        # Add the next node to the route
        route.append(next_node)
        visited.add(next_node)
        distance = adjacency_matrix[current_node, next_node]
        total_distance += distance
        route_distances.append(distance)
        current_node = next_node
        
        # Remove the node from daily goals if it's part of the goals
        if next_node in daily_goals:
            daily_goals.remove(next_node)
    
    return route, total_distance, route_distances

# Example usage with a weighted adjacency matrix reflecting various factors

# Adjacency matrix example for testing(representing distances in miles):
# Nodes: 0: Home, 1: Work, 2: Grocery Store, 3: Childcare, 4: Gym
adjacency_matrix = np.array([
    [0, 5, 3, 2, 7],  # Home
    [5, 0, 4, 4, 2],  # Work
    [3, 4, 0, 6, 1],  # Grocery Store
    [2, 4, 6, 0, 5],  # Childcare
    [7, 2, 1, 5, 0]   # Gym
])

# Personalized vector example:
# Reflecting preferences (higher values for nodes that align with today's goals)
# Testing using today's goals that will prioritize routes to Grocery Store and Childcare
personalized_vector = np.array([0.1, 0.1, 0.4, 0.4, 0])

# Compute DayRank scores
day_rank_scores = day_rank(adjacency_matrix, personalized_vector, teleportation_probability=0.15, max_iterations=100, tol=1e-6)

# Define the starting node and daily goals
start_node = 0  # Home
daily_goals = [2, 3]  # Grocery Store (2) and Childcare (3)

# Find the best route based on DayRank scores and daily goals
route, total_distance, route_distances = find_best_route(adjacency_matrix, day_rank_scores, start_node, daily_goals)

# Print the results
node_names = ["Home", "Work", "Grocery Store", "Childcare", "Gym"]
print("Best Route:")
for i in range(len(route)):
    node = route[i]
    if i < len(route) - 1:
        print(f"{node_names[node]} -> ({route_distances[i]:.2f} miles) -> ", end="")
    else:
        print(f"{node_names[node]}")
print(f"Total Distance: {total_distance:.2f} miles")
