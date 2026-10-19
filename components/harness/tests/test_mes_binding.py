"""Stable AAS linkage resolves current MES IDs even after an order moves."""

import pytest

from a4s.machines.festo_cp_lab.mes_binding import current_demo_orders, resolve_lines
from a4s.mesctl.client import MesError


class Mes:
    config = type("Config", (), {"endpoint": "http://mes.test"})()

    def __init__(self):
        self.lines = [{"id": 8, "code": "A4S-FESTO"}, {"id": 9, "code": "A4S-BACKUP"}]
        self.orders = [
            {"id": 11, "order_no": "A4S-old-A", "line_id": 8},
            {"id": 12, "order_no": "A4S-old-B", "line_id": 8},
            {"id": 30, "order_no": "A4S-new-A", "line_id": 9},
            {"id": 31, "order_no": "A4S-new-B", "line_id": 8},
            {"id": 99, "order_no": "OTHER-X", "line_id": 8},
        ]

    def rows(self, path):
        return self.lines if path == "/api/v1/lines" else self.orders


def test_resolves_live_line_and_latest_pair_after_move():
    mes = Mes()
    context = {"MESBaseUrl": "http://mes.test", "MESLineCode": "A4S-FESTO",
               "MESBackupLineCode": "A4S-BACKUP"}
    line, backup = resolve_lines(context, mes)
    assert [order["id"] for order in current_demo_orders(mes, line, backup)] == [30, 31]
    mes.lines[0]["id"] = 108
    mes.orders[3]["line_id"] = 108
    assert resolve_lines(context, mes)[0]["id"] == 108


def test_rejects_mismatched_mes_endpoint():
    with pytest.raises(MesError, match="trusted MES"):
        resolve_lines({"MESBaseUrl": "http://untrusted", "MESLineCode": "A4S-FESTO",
                       "MESBackupLineCode": "A4S-BACKUP"}, Mes())
