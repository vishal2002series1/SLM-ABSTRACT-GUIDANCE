"""Codebase-level feature task: add 'discount codes' across a multi-file mini-app.

Unlike the single-file sandbox tasks, this requires coherent edits spanning three
files (a new model, a storage registry, and service methods) whose interfaces must
line up. It probes: can the SLM reason ACROSS files, where does Opus help, and is
context selection needed once a repo no longer fits trivially in one prompt.

The starting repo is a *working* order system with NO discount support. The test
suite defines the feature to add. `test_basic_total` already passes; the three
discount tests fail until the feature is implemented across all three files.
"""

CODEBASE_FEATURE_TASK = {
    "id": "feature_discount_codes",
    "feature_name": "Discount codes",
    "description": (
        "Add discount-code support to the order system. Implement: "
        "OrderService.register_discount(code, percent, min_subtotal=0.0) to register a code; "
        "OrderService.apply_discount(order_id, code) which applies the discount to an order, "
        "raising ValueError for an unknown code or when the order subtotal is below the "
        "discount's min_subtotal; and OrderService.order_total(order_id) must return the "
        "subtotal reduced by the applied discount percentage. This spans the model layer "
        "(a Discount type + per-order discount state), the storage layer (a discount registry), "
        "and the service layer (the new methods + total calculation)."
    ),
    "test_file": "test_suite.py",
    # Files the agent is allowed to modify (the test file is excluded == ground truth).
    "editable_files": ["models.py", "storage.py", "service.py"],
    "files": {
        "models.py": '''from dataclasses import dataclass, field
from typing import List


@dataclass
class LineItem:
    name: str
    price: float
    qty: int


@dataclass
class Order:
    id: int
    items: List[LineItem] = field(default_factory=list)
''',
        "storage.py": '''from models import Order


class OrderStore:
    def __init__(self):
        self._orders = {}
        self._next_id = 1

    def create(self, items):
        order = Order(id=self._next_id, items=list(items))
        self._orders[order.id] = order
        self._next_id += 1
        return order

    def get(self, order_id):
        return self._orders[order_id]
''',
        "service.py": '''from storage import OrderStore
from models import LineItem


class OrderService:
    def __init__(self):
        self.store = OrderStore()

    def place_order(self, items):
        return self.store.create([LineItem(n, p, q) for (n, p, q) in items])

    def _subtotal(self, order):
        return sum(i.price * i.qty for i in order.items)

    def order_total(self, order_id):
        order = self.store.get(order_id)
        return self._subtotal(order)
''',
        "test_suite.py": '''import pytest
from service import OrderService


def test_basic_total():
    svc = OrderService()
    o = svc.place_order([("widget", 10.0, 2), ("gadget", 5.0, 1)])
    assert svc.order_total(o.id) == 25.0


def test_apply_percentage_discount():
    svc = OrderService()
    svc.register_discount("SAVE10", percent=10)
    o = svc.place_order([("widget", 10.0, 2)])
    svc.apply_discount(o.id, "SAVE10")
    assert svc.order_total(o.id) == 18.0


def test_minimum_subtotal_not_met():
    svc = OrderService()
    svc.register_discount("BIG20", percent=20, min_subtotal=100.0)
    o = svc.place_order([("widget", 10.0, 1)])
    with pytest.raises(ValueError):
        svc.apply_discount(o.id, "BIG20")
    assert svc.order_total(o.id) == 10.0


def test_unknown_code_rejected():
    svc = OrderService()
    o = svc.place_order([("widget", 10.0, 1)])
    with pytest.raises(ValueError):
        svc.apply_discount(o.id, "NOPE")
''',
    },
}
