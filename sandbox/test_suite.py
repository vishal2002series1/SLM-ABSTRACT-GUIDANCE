def test_dag_scheduler():
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
