from collections import defaultdict, deque

def schedule_tasks(tasks: list, dependencies: list) -> list:
    """
    Performs a topological sort to determine a valid execution order for tasks
    based on given dependencies (prerequisite must run before dependent task).
    
    Raises ValueError if a cycle is detected in the dependencies.
    """
    
    # 1. Initialize Graph and In-Degrees
    graph = defaultdict(list)
    in_degree = {task: 0 for task in tasks}
    
    # Ensure all tasks are accounted for, even if they have no dependencies listed
    # This step is technically redundant if we initialize in_degree correctly, 
    # but keeping it for clarity/safety.
    for task in tasks:
        if task not in in_degree:
            in_degree[task] = 0

    # 2. Build Graph and Calculate In-Degrees
    for prerequisite, task in dependencies:
        # Check if both nodes exist in the provided task list
        if prerequisite not in in_degree or task not in in_degree:
            # Skip dependencies involving unknown tasks
            continue
            
        # prerequisite -> task (prerequisite must run before task)
        graph[prerequisite].append(task)
        in_degree[task] += 1
        
    # 3. Initialize Queue (Tasks with no prerequisites)
    queue = deque([task for task in tasks if in_degree[task] == 0])
    
    # 4. Process (Kahn's Algorithm)
    order = []
    
    while queue:
        current_task = queue.popleft()
        order.append(current_task)
        
        # Process neighbors (tasks that depend on current_task)
        for neighbor in graph[current_task]:
            # Reduce the dependency count for the neighbor
            in_degree[neighbor] -= 1
            
            # If the neighbor now has no remaining prerequisites, it can be scheduled
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    # 5. Cycle Detection
    if len(order) != len(tasks):
        # If the number of tasks in the order is less than the total number of tasks,
        # it means there are remaining tasks that could not be scheduled, indicating a cycle.
        raise ValueError("Cycle detected: Cannot schedule tasks due to circular dependencies.")
        
    return order