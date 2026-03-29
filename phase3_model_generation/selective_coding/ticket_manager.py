"""
Selective Coding – Ticket Manager.

Tickets are strictly item-scoped. Selective coding may request
targeted re-analysis for specific items but does NOT re-run open or
axial phases globally.

Two ticket categories (Algorithm 3):
  1. Structural tickets: circular inheritance, invalid multiplicities,
     unimplemented interfaces, dangling/duplicate items
  2. Coverage tickets: entities/relationships with low evidence coverage

Ticket lifecycle: open -> resolved | unresolved
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TicketStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class TicketCategory(str, Enum):
    STRUCTURAL = "structural"
    COVERAGE = "coverage"


class IssueType(str, Enum):
    CIRCULAR_INHERITANCE = "circular_inheritance"
    INVALID_MULTIPLICITY = "invalid_multiplicity"
    UNIMPLEMENTED_INTERFACE = "unimplemented_interface"
    DANGLING_REFERENCE = "dangling_reference"
    DUPLICATE_ELEMENT = "duplicate_element"
    LOW_COVERAGE = "low_coverage"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class Ticket:
    ticket_id: str
    category: TicketCategory
    issue_type: IssueType
    affected_elements: List[str]          # entity names or relationship keys
    issue_description: str
    status: TicketStatus = TicketStatus.OPEN
    fix_applied: Optional[str] = None     # description of applied fix


class TicketManager:
    def __init__(self):
        self._tickets: Dict[str, Ticket] = {}

    def create(
        self,
        category: TicketCategory,
        issue_type: IssueType,
        affected_elements: List[str],
        description: str,
    ) -> Ticket:
        tid = f"ticket_{uuid.uuid4().hex[:8]}"
        ticket = Ticket(
            ticket_id=tid,
            category=category,
            issue_type=issue_type,
            affected_elements=affected_elements,
            issue_description=description,
        )
        self._tickets[tid] = ticket
        return ticket

    def open_tickets(self) -> List[Ticket]:
        return [t for t in self._tickets.values() if t.status == TicketStatus.OPEN]

    def unresolved_tickets(self) -> List[Ticket]:
        return [t for t in self._tickets.values() if t.status == TicketStatus.UNRESOLVED]

    def resolve(self, ticket: Ticket, fix_description: str) -> None:
        ticket.status = TicketStatus.RESOLVED
        ticket.fix_applied = fix_description

    def mark_unresolved(self, ticket: Ticket) -> None:
        ticket.status = TicketStatus.UNRESOLVED

    def pending_count(self) -> int:
        return len([t for t in self._tickets.values()
                    if t.status in (TicketStatus.OPEN, TicketStatus.UNRESOLVED)])

    def clear_resolved(self) -> None:
        self._tickets = {
            tid: t for tid, t in self._tickets.items()
            if t.status != TicketStatus.RESOLVED
        }

    def all_tickets(self) -> List[Ticket]:
        return list(self._tickets.values())
