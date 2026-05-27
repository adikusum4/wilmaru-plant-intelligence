"""
Alert Engine — centralised severity routing.
Semua modul push alert ke sini; engine memutuskan channel (email/Slack/log).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Severity = Literal["INFO", "WARNING", "CRITICAL"]

@dataclass
class Alert:
    module: str
    severity: Severity
    title: str
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    resolved: bool = False

_queue: list[Alert] = []

def push(module: str, severity: Severity, title: str, detail: str) -> Alert:
    a = Alert(module=module, severity=severity, title=title, detail=detail)
    _queue.append(a)
    icon = {"INFO":"ℹ️","WARNING":"⚠️","CRITICAL":"🚨"}[severity]
    print(f"{icon} [{a.timestamp}] [{module}] {title}")
    return a

def get_all(severity: Severity | None = None) -> list[Alert]:
    if severity: return [a for a in _queue if a.severity == severity]
    return list(_queue)

def clear(): _queue.clear()
