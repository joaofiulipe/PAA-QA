"""
package.py - Passo 5b: prepara o dataset completo para distribuicao ao grupo.

  - compacta cada data\\processed\\*.jsonl em data\\release\\data\\*.jsonl.gz (em paralelo);
  - calcula SHA-256 de cada arquivo compactado;
  - gera data\\release\\manifest.json e data\\release\\README.md (dataset card, compativel
    com Hugging Face);
  - com --zip, gera tambem data\\pubmed_rag_unb.zip (arquivo unico para Google Drive/OneDrive).

Uso (apos o pipeline e o sample.py):
    python scripts\\package.py
    python scripts\\package.py --zip
"""
import argparse
import gzip
import hashlib
import json
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
REL_DIR = ROOT / "data" / "release"
REL_DATA = REL_DIR / "data"
LOG_DIR = ROOT / "logs"
ZIP_PATH = ROOT / "data" / "pubmed_rag_unb.zip"
CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def compress(src_str):
    """Compacta um .jsonl (gzip nivel 6, mtime=0 para ser reprodutivel) e conta as linhas."""
    src = Path(src_str)
    dst = REL_DATA / (src.name + ".gz")
    tmp = dst.with_name(dst.name + ".part")
    lines = 0
    with open(src, "rb") as fin, open(tmp, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as gz:
        while chunk := fin.read(CHUNK):
            lines += chunk.count(b"\n")
            gz.write(chunk)
    tmp.replace(dst)
    return {"file": f"data/{dst.name}", "docs": lines, "uncompressed_bytes": src.stat().st_size,
            "compressed_bytes": dst.stat().st_size, "sha256": sha256(dst)}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


README_TEMPLATE = """---
configs:
- config_name: default
  data_files: "data/*.jsonl.gz"
language:
- en
task_categories:
- question-answering
tags:
- biomedical
- pubmed
- rag
---

# PubMed RAG UnB - titulos e resumos do PubMed (texto limpo + metadados)

Dataset preparado para o projeto de QA com arquitetura RAG (UnB). Contem titulos e resumos
em ingles do **PubMed Baseline {year}** (NLM/NCBI), limpos e estruturados, com metadados
de origem para citacao de fontes.

## Numeros

| Item | Valor |
|---|---|
| Documentos | {docs:,} |
| Texto puro (titulo + resumo, UTF-8) | {text_gb:.2f} GB |
| JSONL descompactado | {jsonl_gb:.2f} GB |
| Compactado (.jsonl.gz) | {gz_gb:.2f} GB em {n_files} arquivos |
| Arquivos de origem do baseline | {src_files} (do mais recente para o mais antigo) |
| PMIDs duplicados | {dups} |

## Esquema (um documento JSON por linha)

| Campo | Tipo | Descricao |
|---|---|---|
| doc_id | str | Identificador unico, `pmid:<PMID>` |
| pmid | str | PubMed ID |
| doi | str ou null | DOI do artigo |
| pmcid | str ou null | ID no PubMed Central (quando houver texto completo aberto) |
| title | str | Titulo do artigo |
| abstract | str | Resumo; secoes estruturadas preservadas como `ROTULO: texto`, separadas por quebra de linha |
| journal | str ou null | Nome do periodico |
| year | int ou null | Ano de publicacao |
| authors | list[str] | Autores ("Nome Sobrenome") ou nomes coletivos |
| mesh_terms | list[str] | Descritores MeSH (vazio em artigos recentes ainda nao indexados) |
| publication_types | list[str] | Tipos de publicacao (Journal Article, Review, ...) |
| language | str | Sempre `eng` |
| source_file | str | Arquivo do baseline de origem |
| url | str | Link de citacao: `https://pubmed.ncbi.nlm.nih.gov/<PMID>/` |

## Limpeza aplicada

Mantidos apenas registros em ingles, com titulo e resumo de pelo menos 200 caracteres, excluindo
publicacoes retratadas e notas de retratacao. Unicode normalizado (NFC), caracteres invisiveis
removidos, espacos colapsados, tags inline achatadas. Caixa, pontuacao e numeros preservados
(sem tokenizacao, stopwords ou lematizacao - isso fica a cargo de cada modelo).
Descartes: {dropped}

## Como carregar

Python puro (streaming, memoria constante):

```python
import glob, gzip, json

for path in sorted(glob.glob("data/*.jsonl.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
```

Hugging Face `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="data/*.jsonl.gz", split="train")
```

Integridade: confira os SHA-256 em `manifest.json`.

## Fonte e uso

Dados da U.S. National Library of Medicine (PubMed Baseline, ftp.ncbi.nlm.nih.gov/pubmed/baseline).
Alguns resumos tem direitos autorais dos editores: uso restrito a fins academicos neste projeto.
Gerado em {created}.
"""


def main():
    ap = argparse.ArgumentParser(description="Empacota o dataset para distribuicao")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--zip", action="store_true", help="gera tambem um .zip unico")
    args = ap.parse_args()

    files = sorted(PROC_DIR.glob("*.jsonl"))
    if not files:
        print("Nenhum .jsonl em data\\processed.")
        return
    REL_DATA.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"Compactando {len(files)} arquivos com {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        entries = list(pool.map(compress, map(str, files)))
    t_gz = time.perf_counter() - t0

    summary = load_json(LOG_DIR / "pipeline_summary.json")
    validation = load_json(LOG_DIR / "validation.json")
    docs = sum(e["docs"] for e in entries)
    gz_bytes = sum(e["compressed_bytes"] for e in entries)
    jsonl_bytes = sum(e["uncompressed_bytes"] for e in entries)
    created = time.strftime("%Y-%m-%d %H:%M:%S")

    manifest = {
        "name": "pubmed-rag-unb",
        "created_at": created,
        "source": "PubMed Baseline 2026 (pubmed26n*.xml.gz), NLM/NCBI",
        "documents": docs,
        "text_bytes": summary.get("text_bytes_total"),
        "jsonl_bytes": jsonl_bytes,
        "compressed_bytes": gz_bytes,
        "duplicate_pmids": validation.get("full_dataset_duplicate_pmids"),
        "compress_seconds": round(t_gz, 1),
        "files": entries,
    }
    (REL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = README_TEMPLATE.format(
        year=2026, docs=docs, text_gb=(summary.get("text_bytes_total") or 0) / 1e9,
        jsonl_gb=jsonl_bytes / 1e9, gz_gb=gz_bytes / 1e9, n_files=len(entries),
        src_files=summary.get("files_processed_total", len(entries)),
        dups=validation.get("full_dataset_duplicate_pmids", "n/d"),
        dropped=summary.get("dropped_total", "n/d"), created=created)
    (REL_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Compactado: {jsonl_bytes / 1e9:.2f} GB -> {gz_bytes / 1e9:.2f} GB "
          f"(razao {jsonl_bytes / max(gz_bytes, 1):.1f}x) em {t_gz:.1f}s")
    print(f"Release pronto em {REL_DIR}")

    if args.zip:
        t1 = time.perf_counter()
        with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            z.write(REL_DIR / "README.md", "README.md")
            z.write(REL_DIR / "manifest.json", "manifest.json")
            for e in entries:
                z.write(REL_DIR / e["file"], e["file"])
        print(f"ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e9:.2f} GB) em {time.perf_counter() - t1:.1f}s")


if __name__ == "__main__":
    main()
