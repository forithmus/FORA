"""Node-local copy of the DPO pair store (paper: "Preference optimisation", CT DPO stage).

Each training rank sees only the train reports whose block-23 DINO cache is staged on its node (data rank
``nodes_rank * gpus + local_rank``, see ``fora.reportgen.writer.dino_stream.local_report_cases``). This tool writes, once per
node, a copy of the global pair store that keeps only the pairs of this node's train reports, so no rank can draw a
pair of a report whose cache lives elsewhere and the node-local file is ~1/32 of the global store. The output path
should be identical on every node; its metadata records node, data ranks, uid hash and source, and a foreign or stale
file is rebuilt.

    python -m fora_report.ct.dpo_pairs_local --source pairs_all.sqlite --output /local/pairs.sqlite \
        --mil-config mil_config.json --index report_index.sqlite --cache-root /local/block23_cache [--node N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from fora_report.ct.dpo_loss import PREFERENCE_SCHEMA

FILTER_VERSION = "node_local_filter"


def node_data_ranks(node: int, ranks_per_node: int = 4) -> list[int]:
    return [int(node) * int(ranks_per_node) + g for g in range(int(ranks_per_node))]


def local_train_uids(config: dict, data_ranks, index_splits: dict, *, assign=None, split: str = "train") -> dict:
    """data rank -> sorted uids of its node-local reports of ``split`` (exactly the trainer's dataset set)."""
    from fora.reportgen.writer.dino_stream import local_report_cases
    A = assign if assign is not None else json.loads(Path(config["assign"]).read_text())
    return {int(r): sorted(local_report_cases(config, int(r), index_splits, split, assign=A)) for r in data_ranks}


def uid_digest(uids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, uids))).encode()).hexdigest()


def source_identity(path) -> dict:
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": int(st.st_size), "mtime": int(st.st_mtime)}


def read_filter_metadata(path) -> dict | None:
    try:
        connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT value FROM metadata WHERE key='local_filter'").fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    return json.loads(row[0]) if row else None


def filter_pair_store(source, output, uids, *, provenance: dict | None = None) -> dict:
    """Write ``output``: the pairs of ``source`` whose report_uid is in ``uids`` (atomic rename).

    Keeps the pair-store schema/summary (pairs_by_type and reports recomputed) and adds a ``local_filter``
    metadata row, which is returned.
    """
    source, output = Path(source), Path(output)
    keep = sorted(set(map(str, uids)))
    tmp = Path(f"{output}.partial.{os.getpid()}")
    tmp.unlink(missing_ok=True)
    out = sqlite3.connect(f"file:{tmp.resolve()}", uri=True)
    try:
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.execute("ATTACH DATABASE ? AS src", (f"file:{source.resolve()}?mode=ro",))
        schema = {name: sql for name, sql in out.execute(
            "SELECT name, sql FROM src.sqlite_master WHERE type IN ('table','index') AND sql IS NOT NULL")}
        for name in ("pairs", "metadata"):
            if name not in schema:
                raise RuntimeError(f"{source}: missing table {name}")
            out.execute(schema[name])
        summary = json.loads(out.execute("SELECT value FROM src.metadata WHERE key='summary'").fetchone()[0])
        if summary.get("schema") != PREFERENCE_SCHEMA:
            raise RuntimeError(f"{source}: not a CT preference index ({summary.get('schema')})")
        out.execute("CREATE TEMP TABLE keep(uid TEXT PRIMARY KEY)")
        out.executemany("INSERT INTO keep VALUES (?)", ((u,) for u in keep))
        out.execute("INSERT INTO main.pairs SELECT p.* FROM src.pairs p JOIN keep k ON p.report_uid = k.uid "
                    "ORDER BY p.report_uid, p.pair_id")
        for name, sql in schema.items():
            if name not in ("pairs", "metadata") and sql.upper().startswith("CREATE INDEX"):
                out.execute(sql)
        by_type = dict(out.execute("SELECT pair_type, COUNT(*) FROM main.pairs GROUP BY pair_type ORDER BY 1"))
        reports = out.execute("SELECT COUNT(DISTINCT report_uid) FROM main.pairs").fetchone()[0]
        filt = {
            "version": FILTER_VERSION, "source": source_identity(source), "n_uids": len(keep),
            "uids_sha256": uid_digest(keep), "pairs": int(sum(by_type.values())), "reports_with_pairs": int(reports),
            "source_pairs_by_type": summary.get("pairs_by_type"), **(provenance or {}),
        }
        summary = dict(summary, pairs_by_type=by_type, reports=int(reports), local_filter=filt)
        for key, value in out.execute("SELECT key, value FROM src.metadata"):
            if key not in ("summary", "local_filter"):
                out.execute("INSERT INTO main.metadata VALUES (?, ?)", (key, value))
        out.execute("INSERT INTO main.metadata VALUES ('summary', ?)", (json.dumps(summary, sort_keys=True),))
        out.execute("INSERT INTO main.metadata VALUES ('local_filter', ?)", (json.dumps(filt, sort_keys=True),))
        out.commit()
        out.execute("DETACH DATABASE src")
    finally:
        out.close()
    os.replace(tmp, output)
    return filt


def ensure_node_store(source, output, config: dict, index: str, node: int, *, assign=None) -> dict:
    """Build (or reuse when identical) the node-local pair store of ``node``."""
    from fora.reportgen.writer.dino_stream import read_index_splits
    A = assign if assign is not None else json.loads(Path(config["assign"]).read_text())
    ranks = node_data_ranks(node, int(A.get("ranks_per_node", 4)))
    by_rank = local_train_uids(config, ranks, read_index_splits(index), assign=A)
    uids = sorted({u for v in by_rank.values() for u in v})
    provenance = {"node": int(node), "data_ranks": ranks, "fingerprint": config["fingerprint"],
                  "index": str(Path(index).resolve()),
                  "uids_per_rank": {str(r): len(v) for r, v in by_rank.items()}}
    current = read_filter_metadata(output) if Path(output).is_file() else None
    if current and current.get("version") == FILTER_VERSION and current.get("node") == int(node) \
            and current.get("uids_sha256") == uid_digest(uids) and current.get("source") == source_identity(source) \
            and current.get("fingerprint") == config["fingerprint"]:
        return dict(current, reused=True)
    return dict(filter_pair_store(source, output, uids, provenance=provenance), reused=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mil-config", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--node", type=int, default=-1, help="data node (default: the staged-cache marker on this host)")
    p.add_argument("--cache-root", required=True, help="root of the node-local block-23 caches (marker lookup)")
    args = p.parse_args(argv)
    from fora.reportgen.writer.dino_stream import detect_data_node
    config = json.loads(Path(args.mil_config).read_text())
    node = args.node if args.node >= 0 else detect_data_node(config["fingerprint"], Path(args.cache_root))
    result = ensure_node_store(args.source, args.output, config, args.index, node)
    keys = ("reused", "n_uids", "pairs", "reports_with_pairs", "data_ranks")
    print(f"[dpo-pairs] node={node} {json.dumps({k: result[k] for k in keys})}", flush=True)


if __name__ == "__main__":
    main()
