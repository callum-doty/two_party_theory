"""
The margin/spending-response model, reused unchanged from the old project's
estimation pipeline (project_spec.md Section 4: "the existing research
already represents race state using expected margin, uncertainty, ...").

This project does not re-estimate the response curve; it re-derives what
counts as STRATEGIC behavior on top of it (game/), per spec Section 18's
identification concern -- see that section for why the old eta reaction
estimate is treated as descriptive evidence here, not as the definition of
strategic response.
"""

from __future__ import annotations

from backtest.model.margin import MarginModelCoefficients, predict_floor_margin  # noqa: F401
from backtest.model.ceiling import ceiling  # noqa: F401
