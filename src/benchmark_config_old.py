BENCHMARK_TASKS = [
    {
        "id": "task_01_lcs",
        "description": "Compute the longest common subsequence of two strings using dynamic programming.",
        "buggy_code": """def longest_common_subsequence(text1: str, text2: str) -> int:
    if not text1 or not text2:
        return 0
    if text1[0] == text2[0]:
        return 1 + longest_common_subsequence(text1[1:], text2[1:])
    else:
        return longest_common_subsequence(text1[1:], text2)
""",
        "test_suite": """def test_lcs():
    from target_app import longest_common_subsequence
    assert longest_common_subsequence("abc", "abc") == 3
    assert longest_common_subsequence("abc", "def") == 0
    assert longest_common_subsequence("abcde", "ace") == 3
    assert longest_common_subsequence("bl", "yby") == 1
"""
    },
    {
        "id": "task_02_levenshtein",
        "description": "Calculate the minimum Levenshtein (edit) distance between two strings.",
        "buggy_code": """def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) == 0: return len(s2)
    if len(s2) == 0: return len(s1)
    if s1[0] == s2[0]:
        return levenshtein_distance(s1[1:], s2[1:])
    return 1 + levenshtein_distance(s1[1:], s2)
""",
        "test_suite": """def test_levenshtein():
    from target_app import levenshtein_distance
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("hello", "hello") == 0
    assert levenshtein_distance("intention", "execution") == 5
"""
    },
    {
        "id": "task_03_transactional_db",
        "description": "Implement an in-memory Key-Value database that supports nested transactions (BEGIN, COMMIT, ROLLBACK).",
        "buggy_code": """class TransactionDB:
    def __init__(self):
        self.data = {}
        self.transaction_stack = []

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key, None)

    def delete(self, key):
        if key in self.data:
            del self.data[key]

    def begin(self):
        # Buggy: Just copying the reference, not creating a deep snapshot for nested rollback
        self.transaction_stack.append(self.data)

    def commit(self):
        if self.transaction_stack:
            self.transaction_stack.pop()

    def rollback(self):
        if self.transaction_stack:
            # Buggy: Fails to properly restore state across nested levels
            self.data = self.transaction_stack.pop()
""",
        "test_suite": """def test_transactional_db():
    from target_app import TransactionDB
    db = TransactionDB()
    
    # Basic CRUD
    db.set("a", 1)
    assert db.get("a") == 1
    
    # Simple Transaction
    db.begin()
    db.set("a", 2)
    assert db.get("a") == 2
    db.rollback()
    assert db.get("a") == 1
    
    # Nested Transactions (This will break the buggy code)
    db.begin()
    db.set("b", 10)
    db.begin()
    db.set("b", 20)
    assert db.get("b") == 20
    db.rollback()
    assert db.get("b") == 10
    db.commit()
    assert db.get("b") == 10
"""
    },
    {
        "id": "task_04_dag_scheduler",
        "description": "Implement a task scheduler that returns task execution order based on dependencies, and detects circular dependencies.",
        "buggy_code": """def schedule_tasks(tasks: list, dependencies: list) -> list:
    # Buggy Implementation: A naive approach that doesn't handle complex trees or cycle detection
    order = []
    for task in tasks:
        if task not in order:
            order.append(task)
            
    # Naively attempting to push dependencies first
    for dep in dependencies:
        parent, child = dep[0], dep[1]
        if parent in order and child in order:
            p_idx = order.index(parent)
            c_idx = order.index(child)
            if p_idx > c_idx:
                # Swap them
                order[p_idx], order[c_idx] = order[c_idx], order[p_idx]
                
    return order
""",
        "test_suite": """def test_dag_scheduler():
    from target_app import schedule_tasks
    import pytest
    
    # Simple linear dependency
    tasks = ["A", "B", "C"]
    deps = [("A", "B"), ("B", "C")] # A must run before B, B before C
    order = schedule_tasks(tasks, deps)
    assert order.index("A") < order.index("B")
    assert order.index("B") < order.index("C")
    
    # Complex tree
    tasks2 = ["Web", "DB", "Cache", "API"]
    deps2 = [("DB", "API"), ("Cache", "API"), ("API", "Web")]
    order2 = schedule_tasks(tasks2, deps2)
    assert order2.index("DB") < order2.index("API")
    assert order2.index("Cache") < order2.index("API")
    assert order2.index("API") < order2.index("Web")
    
    # Cycle Detection (Must raise ValueError)
    tasks3 = ["A", "B"]
    deps3 = [("A", "B"), ("B", "A")]
    with pytest.raises(ValueError):
        schedule_tasks(tasks3, deps3)
"""
    }
]