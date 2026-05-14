"""Parent-child chunk relationship adapters."""

from __future__ import annotations

from dataclasses import dataclass

from .child_chunks import ChildChunk
from .parent_chunk import ParentRecord


@dataclass(frozen=True)
class ParentChildRelation:
    parent: ParentRecord
    child: ChildChunk


class ParentChildRelationshipAdapter:
    """Relate generated child chunks back to their parent record."""

    def relate(
        self,
        parent: ParentRecord,
        children: list[ChildChunk],
    ) -> list[ParentChildRelation]:
        relations: list[ParentChildRelation] = []
        seen_child_ids: set[str] = set()

        for child in children:
            if child.chunk_id in seen_child_ids:
                raise ValueError(f"Duplicate child chunk ID found: {child.chunk_id}")

            if child.parent_record_id != parent.record_id:
                raise ValueError(
                    f"Child {child.chunk_id} references parent "
                    f"{child.parent_record_id}, but expected {parent.record_id}."
                )

            if child.parent.record_id != parent.record_id:
                raise ValueError(
                    f"Child {child.chunk_id} contains parent object "
                    f"{child.parent.record_id}, but expected {parent.record_id}."
                )

            seen_child_ids.add(child.chunk_id)
            relations.append(ParentChildRelation(parent=parent, child=child))

        return relations