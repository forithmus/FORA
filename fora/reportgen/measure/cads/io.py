"""Input helpers shared by the CADS anatomy measurers (paper: Methods, "Measurements").

The CT volumes are read from the CT-FORA H5 store and the CADS label volumes from one `cads_<h5>.h5` file per H5 batch.
Both roots are set once per process with `set_roots()` (CLI flags `--h5-root` / `--cads-root`).
A case-list line (see `caselist.py`) gives the provenance of every CT series of a study: `{case, report, h5, patient,
series: [[accession, series_id], ...], classes}`.
"""
import os

ROOTS = dict(h5='', cads='')


def set_roots(h5_root: str, cads_root: str) -> None:
    ROOTS['h5'] = str(h5_root); ROOTS['cads'] = str(cads_root)


def cads_file(h5_id: str) -> str:
    return os.path.join(ROOTS['cads'], f'cads_{h5_id}.h5')


def h5_file(h5_id: str) -> str:
    return os.path.join(ROOTS['h5'], f'{h5_id}.h5')


def cads_group(f, p, sid):
    """CADS group of one series inside a cads_<h5>.h5 file (None if the series was not segmented)."""
    k = f"{p['patient']}/{p['report']}/{p['acc']}/{sid}"
    return f[k] if k in f else None


def case_series(c):
    """(provenance, series meta) per series from the case-list line (no feature pack needed)."""
    for acc, sid in c['series']:
        yield dict(h5=c['h5'], patient=c['patient'], report=c['report'], acc=acc, sid=sid, sub='', frame=0), dict(relative_file=f'{acc}/{sid}')
