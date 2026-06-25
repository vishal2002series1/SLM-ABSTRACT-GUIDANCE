COMPLEX_BENCHMARK_TASKS = [
    {
        "id": "task_06_token_bucket_rate_limiter",
        "description": "Implement a multi-tenant Token Bucket Rate Limiter across a distributed-style interface. The system must support thread-safe token consumption, bucket refilling based on real-time millisecond deltas, and handle burst windows accurately without drifting state.",
        "files": {
            "limiter_core.py": """import time

class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens: float) -> bool:
        # Buggy: Missing time-delta calculation based on float milliseconds, failing burst tests
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
""",
            "test_suite.py": """import time
import pytest
from limiter_core import TokenBucket

def test_rate_limiting_burst():
    bucket = TokenBucket(capacity=10.0, refill_rate=2.0) # 2 tokens per second
    assert bucket.consume(5) is True
    assert bucket.consume(6) is False # Insufficient tokens
    
def test_rate_limiting_refill():
    bucket = TokenBucket(capacity=5.0, refill_rate=5.0)
    assert bucket.consume(5) is True
    time.sleep(0.5) # Refills 2.5 tokens
    assert bucket.consume(2) is True
    assert bucket.consume(1) is False
"""
        }
    },
    {
        "id": "task_07_pubsub_broker",
        "description": "Implement an asynchronous Event Broker supporting wildcard topic subscriptions (e.g., 'sport.*.score' matching 'sport.football.score'). It must handle dead-letter queuing (DLQ) for events that fail to deliver after 3 routing attempts.",
        "files": {
            "broker.py": """class EventBroker:
    def __init__(self):
        self.subscriptions = {}
        self.dlq = []

    def subscribe(self, topic_pattern: str, subscriber_id: str):
        self.subscriptions[topic_pattern] = subscriber_id

    def publish(self, topic: str, payload: dict) -> int:
        # Buggy: Doing direct string matching instead of handling regex/wildcard splits
        # Also entirely missing retry tracking for DLQ routing
        delivered = 0
        if topic in self.subscriptions:
            delivered += 1
        return delivered
""",
            "test_suite.py": """import pytest
from broker import EventBroker

def test_wildcard_routing():
    b = EventBroker()
    b.subscribe("orders.*.processed", "sub1")
    assert b.publish("orders.us.processed", {"id": 1}) == 1
    assert b.publish("orders.eu.failed", {"id": 2}) == 0

def test_dead_letter_queue():
    b = EventBroker()
    # Mocking broken subscription path that forces retries
    b.subscribe("fail.topic", "broken_sub")
    # Expected behavior: If subscriber drops exception 3 times, route to b.dlq
    # The current buggy implementation doesn't support DLQ metadata tracking
    assert len(b.dlq) == 0
"""
        }
    },
    {
        "id": "task_08_expression_evaluator",
        "description": "Implement a recursive-descent arithmetic expression evaluator supporting + - * / operators with correct precedence and associativity, parenthesized subexpressions, and unary minus. The current implementation evaluates strictly left-to-right and ignores operator precedence, parentheses, and unary negation.",
        "files": {
            "evaluator.py": """class Evaluator:
    def evaluate(self, expr: str) -> float:
        # Buggy: naive left-to-right scan, no precedence, no parens, no unary minus.
        tokens = expr.replace(' ', '')
        total = 0
        num = ''
        op = '+'
        for ch in tokens:
            if ch.isdigit():
                num += ch
            else:
                if op == '+':
                    total += int(num)
                num = ''
                op = ch
        total += int(num)
        return total
""",
            "test_suite.py": """import pytest
from evaluator import Evaluator

def test_precedence():
    e = Evaluator()
    assert e.evaluate("2+3*4") == 14

def test_parentheses():
    e = Evaluator()
    assert e.evaluate("(2+3)*4") == 20

def test_division_and_subtraction():
    e = Evaluator()
    assert e.evaluate("10/2-3") == 2

def test_unary_minus():
    e = Evaluator()
    assert e.evaluate("-5+3") == -2
    assert e.evaluate("2*-3") == -6

def test_nested():
    e = Evaluator()
    assert e.evaluate("((1+2)*(3+4))") == 21
"""
        }
    },
    {
        "id": "task_09_topological_sort",
        "description": "Implement a deterministic topological sort. `topo_sort(graph)` takes a dict mapping each node to the list of nodes it depends on (must come before it), and returns a valid linear ordering. When multiple nodes are simultaneously available, the lexicographically smallest must be emitted first. If the graph contains a cycle, raise ValueError('cycle detected'). The current implementation just returns the keys unsorted and never detects cycles.",
        "files": {
            "scheduler.py": """class Scheduler:
    def topo_sort(self, graph: dict) -> list:
        # Buggy: returns insertion order, ignores dependencies, no cycle detection,
        # no deterministic tie-breaking.
        return list(graph.keys())
""",
            "test_suite.py": """import pytest
from scheduler import Scheduler

def test_linear_chain():
    s = Scheduler()
    # c depends on b, b depends on a  ->  a, b, c
    g = {"c": ["b"], "b": ["a"], "a": []}
    assert s.topo_sort(g) == ["a", "b", "c"]

def test_lexicographic_tiebreak():
    s = Scheduler()
    # d depends on both b and c; b and c depend on a.
    # Available-set order must break ties lexicographically.
    g = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
    assert s.topo_sort(g) == ["a", "b", "c", "d"]

def test_independent_nodes_sorted():
    s = Scheduler()
    g = {"z": [], "a": [], "m": []}
    assert s.topo_sort(g) == ["a", "m", "z"]

def test_cycle_raises():
    s = Scheduler()
    g = {"a": ["b"], "b": ["a"]}
    with pytest.raises(ValueError):
        s.topo_sort(g)
"""
        }
    },
    {
        "id": "task_10_glob_matcher",
        "description": "Implement a shell-style glob matcher. `matches(pattern, text)` returns True iff the whole text matches the pattern. Supported wildcards: '*' matches any sequence of characters (including empty) and requires backtracking; '?' matches exactly one character; '[abc]' matches one character in the set; '[a-z]' matches one character in the inclusive range; '[!...]' negates the class. Matching must be anchored (full string). The current implementation does naive character-by-character equality and treats wildcards as literals.",
        "files": {
            "matcher.py": """class GlobMatcher:
    def matches(self, pattern: str, text: str) -> bool:
        # Buggy: treats every pattern character as a literal; no wildcard handling,
        # no backtracking for '*', no '?' / character-class / negation support.
        if len(pattern) != len(text):
            return False
        for pc, tc in zip(pattern, text):
            if pc != tc:
                return False
        return True
""",
            "test_suite.py": """import pytest
from matcher import GlobMatcher

def test_literal():
    m = GlobMatcher()
    assert m.matches("abc", "abc") is True
    assert m.matches("abc", "abd") is False

def test_question_mark():
    m = GlobMatcher()
    assert m.matches("a?c", "abc") is True
    assert m.matches("a?c", "ac") is False

def test_star_backtracking():
    m = GlobMatcher()
    assert m.matches("a*b", "axyzb") is True
    assert m.matches("*.txt", "report.txt") is True
    assert m.matches("a*a", "aa") is True
    assert m.matches("*a", "aa") is True
    assert m.matches("a*", "a") is True
    assert m.matches("a*b", "axyz") is False

def test_char_class():
    m = GlobMatcher()
    assert m.matches("[abc]at", "bat") is True
    assert m.matches("[a-z]at", "zat") is True
    assert m.matches("[a-z]at", "1at") is False

def test_negated_class():
    m = GlobMatcher()
    assert m.matches("[!0-9]bc", "abc") is True
    assert m.matches("[!0-9]bc", "1bc") is False

def test_combined():
    m = GlobMatcher()
    assert m.matches("src/*/[a-z]?.py", "src/lib/a1.py") is True
    assert m.matches("src/*/[a-z]?.py", "src/lib/A1.py") is False
"""
        }
    }
]