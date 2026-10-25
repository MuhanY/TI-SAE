"""External backbones ported from published work.

Each module here is a *byte-for-byte port* of an upstream backbone implementation
with a small number of *surgical edits* clearly listed in the file header.
The surgical edits are required to:
  - drop dependencies on the source repo's `models.py` / `utils.py` / `Logger` /
    `argparse` glue (replaced with `tisae.models.external.heads.TimeSeriesModelBase`)
  - add a `get_contextual_embeddings(...)` method that returns the per-event
    `[B, L, hid_dim]` representation BEFORE pooling, so the
    :class:`tisae.models.backbone_adapter.BackboneAdapter` can scatter it into
    the (T, V) grid that TISAE expects
  - drop the demographics concat (no demographics in any backbone)
  - apply any one-line bug fixes documented in the per-file header

Each ported file MUST list its surgical edits in a top-of-file comment block so
that future maintainers can re-port from upstream cleanly.
"""

from tisae.models.external.heads import TimeSeriesModelBase  # noqa: F401
