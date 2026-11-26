"""
sample.py - Passo 5a: valida o dataset completo e gera a amostra (<100 MB) para o Aprender3.

Uma unica passada sobre todos os data\\processed\\*.jsonl:
  - conta documentos e verifica PMIDs duplicados entre arquivos (duplicatas nao entram na amostra);
  - amostragem de Bernoulli com semente fixa: cada documento entra com probabilidade p,
    calculada para atingir ~target-mb; teto rigido em hard-cap-mb.
Complexidade: tempo O(N) no total de linhas; memoria O(D) (set de PMIDs como int).

Uso (apos o pipeline terminar):
    python scripts\\sample.py                  # alvo 95 MB, semente 42
    python scripts\\sample.py --target-mb 90 --seed 7
"""
import argparse
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"
LOG_DIR = ROOT / "logs"
PREFIX = b'{"doc_id": "pmid:'


def main():
    ap = argparse.ArgumentParser(description="Validacao + amostra para o Aprender3")
    ap.add_argument("--target-mb", type=float, default=95.0)
    ap.add_argument("--hard-cap-mb", type=float, default=99.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(PROC_DIR.glob("*.jsonl"))
    if not files:
        print("Nenhum .jsonl em data\\processed.")
        return
    total_bytes = sum(f.stat().st_size for f in files)
    p = min(1.0, args.target_mb * 1e6 / total_bytes)
    hard_cap = args.hard_cap_mb * 1e6
    rng = random.Random(args.seed)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / f"pubmed_sample_seed{args.seed}.jsonl"
    tmp_path = out_path.with_name(out_path.name + ".part")

    seen = set()
    docs = dups = s_docs = s_bytes = s_text = 0
    years = {}
    t0 = time.perf_counter()

    print(f"Lendo {len(files)} arquivos ({total_bytes / 1e9:.2f} GB) | p = {p:.6f} | semente {args.seed}")
    with open(tmp_path, "wb") as out:
        for f in files:
            with open(f, "rb") as fin:
                for line in fin:
                    docs += 1
                    pmid = int(line[len(PREFIX):line.index(b'"', len(PREFIX))])
                    if pmid in seen:
                        dups += 1
                        continue
                    seen.add(pmid)
                    if rng.random() < p and s_bytes + len(line) <= hard_cap:
                        out.write(line)
                        s_docs += 1
                        s_bytes += len(line)
                        d = json.loads(line)
                        s_text += len(d["title"].encode("utf-8")) + len(d["abstract"].encode("utf-8"))
                        years[d["year"]] = years.get(d["year"], 0) + 1
    tmp_path.replace(out_path)
    elapsed = time.perf_counter() - t0

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "probability": p,
        "method": "Bernoulli sampling over all documents (uniform, reproducible)",
        "sample_file": out_path.name,
        "sample_docs": s_docs,
        "sample_file_bytes": s_bytes,
        "sample_text_bytes": s_text,
        "sample_year_distribution": dict(sorted(years.items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "full_dataset_docs": docs,
        "full_dataset_unique_pmids": len(seen),
        "full_dataset_duplicate_pmids": dups,
        "full_dataset_jsonl_bytes": total_bytes,
        "seconds": round(elapsed, 1),
    }
    (SAMPLE_DIR / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (LOG_DIR / "validation.json").write_text(json.dumps(
        {k: v for k, v in manifest.items() if k.startswith("full_dataset")}, indent=2), encoding="utf-8")

    print(f"\nDataset completo: {docs:,} docs | {len(seen):,} PMIDs unicos | {dups:,} duplicados")
    print(f"Amostra: {s_docs:,} docs | arquivo {s_bytes / 1e6:.1f} MB | texto puro {s_text / 1e6:.1f} MB")
    print(f"Salvo em {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
