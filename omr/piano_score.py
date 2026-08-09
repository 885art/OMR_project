"""Small MusicXML layout helpers for two-staff piano scores."""

from __future__ import annotations

from typing import Any


def apply_piano_staff_group(score: Any) -> bool:
    """Group exactly two exported parts under one piano brace.

    This changes score semantics/layout only; it does not claim support for
    cross-staff beams or voice separation.  Returning ``False`` lets callers
    record that a malformed non-two-staff result was left untouched.
    """

    from music21 import instrument, layout

    parts = list(score.parts)
    if len(parts) != 2:
        return False
    for part in parts:
        if part.getInstrument(returnDefault=False) is None:
            part.insert(0, instrument.Piano())
    group = layout.StaffGroup(
        parts,
        name="Piano",
        abbreviation="Pno.",
        symbol="brace",
        barTogether=True,
    )
    score.insert(0, group)
    return True
