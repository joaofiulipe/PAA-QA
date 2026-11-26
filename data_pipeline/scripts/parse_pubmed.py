"""
parse_pubmed.py - Passo 3: extrai, limpa e estrutura titulo + resumo do PubMed XML.

Para cada data\\raw\\pubmed26nXXXX.xml.gz gera data\\processed\\pubmed26nXXXX.jsonl
(um documento por linha, UTF-8) e registra estatisticas em logs\\parse_stats.jsonl.

Uso (a partir de C:\\pubmed-rag, com o venv ativo):
    python scripts\\parse_pubmed.py                         # processa todos os .gz em data\\raw
    python scripts\\parse_pubmed.py --workers 4 --delete-raw  # apaga cada .gz apos sucesso

Complexidade (n = bytes de entrada, D = documentos, p = workers):
    tempo  O(n)  (cada byte lido/parseado uma vez; ~O(n/p) em tempo de parede)
    memoria O(1) por worker em relacao a n (iterparse + clear: so 1 registro em memoria)
             + O(D_arquivo) para o set de PMIDs de deduplicacao dentro do arquivo
"""
import argparse
import gzip
import json
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import psutil
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"
STATS_LOG = LOG_DIR / "parse_stats.jsonl"
DONE_FILE = LOG_DIR / "processed.txt"

MIN_ABSTRACT_CHARS = 200
KEEP_LANGS = {"eng"}
EXCLUDE_PUBTYPES = {"Retracted Publication", "Retraction of Publication"}
IGNORED_LABELS = {"UNLABELLED", "UNASSIGNED", ""}
YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")
MEM_SAMPLE_EVERY = 2000


# ---------------------------------------------------------------- limpeza

# caracteres invisiveis: espaco de largura zero vira espaco; os demais sao removidos
ZERO_WIDTH = {0x200B: " ", 0x200C: None, 0x200D: None, 0x2060: None, 0xFEFF: None, 0x00AD: None}
# tags que costumam marcar titulos de secao dentro do texto (ex.: <b>IMPORTANCE</b>)
SPACED_TAGS = {"b", "u"}


def clean(s):
    """Normaliza Unicode (NFC), remove caracteres invisiveis e colapsa espacos.
    Preserva caixa e pontuacao."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s).translate(ZERO_WIDTH)
    return " ".join(s.split())


def flat_text(el):
    """Achata o elemento. <i>/<sup>/<sub> sao colados (H<sub>2</sub>O -> H2O);
    <b>/<u> ganham espacos, e titulos em caixa alta viram 'TITULO:'."""
    parts = [el.text or ""]
    for child in el:
        inner = flat_text(child)
        if child.tag in SPACED_TAGS:
            inner = inner.strip()
            if inner.isupper() and len(inner) <= 40 and not inner.endswith(":"):
                inner += ":"
            inner = f" {inner} "
        parts.append(inner)
        parts.append(child.tail or "")
    return "".join(parts)


def text_of(el):
    """Texto completo e limpo do elemento."""
    return clean(flat_text(el)) if el is not None else ""


# ---------------------------------------------------------------- extracao

def get_year(article):
    pd = article.find("Journal/JournalIssue/PubDate")
    if pd is not None:
        raw = pd.findtext("Year") or pd.findtext("MedlineDate") or ""
        m = YEAR_RE.search(raw)
        if m:
            return int(m.group(1))
    return None


def get_authors(article):
    authors = []
    for a in article.iterfind("AuthorList/Author"):
        coll = a.findtext("CollectiveName")
        if coll:
            authors.append(clean(coll))
            continue
        last = a.findtext("LastName")
        fore = a.findtext("ForeName") or a.findtext("Initials") or ""
        if last:
            authors.append(clean(f"{fore} {last}"))
    return authors


def get_abstract(article):
    parts = []
    for at in article.iterfind("Abstract/AbstractText"):
        t = text_of(at)
        if not t:
            continue
        label = clean(at.get("Label") or "")
        parts.append(t if label.upper() in IGNORED_LABELS else f"{label}: {t}")
    return "\n".join(parts)


def parse_record(pa, source_file):
    """Recebe um <PubmedArticle>. Retorna (documento, None) ou (None, motivo_do_descarte)."""
    mc = pa.find("MedlineCitation")
    art = mc.find("Article") if mc is not None else None
    if art is None:
        return None, "no_article"

    langs = {l.text.strip() for l in art.iterfind("Language") if l.text}
    if not langs & KEEP_LANGS:
        return None, "non_english"

    pubtypes = [clean(p.text) for p in art.iterfind("PublicationTypeList/PublicationType") if p.text]
    if EXCLUDE_PUBTYPES & set(pubtypes):
        return None, "retracted"

    abstract = get_abstract(art)
    if not abstract:
        return None, "no_abstract"
    if len(abstract) < MIN_ABSTRACT_CHARS:
        return None, "short_abstract"

    title = text_of(art.find("ArticleTitle"))
    if title.startswith("[") and title.endswith("]."):  # titulos traduzidos vem entre colchetes
        title = title[1:-2] + "."
    if not title:
        return None, "no_title"

    pmid = clean(mc.findtext("PMID"))
    ids = {aid.get("IdType"): clean(aid.text)
           for aid in pa.iterfind("PubmedData/ArticleIdList/ArticleId") if aid.text}

    doc = {
        "doc_id": f"pmid:{pmid}",
        "pmid": pmid,
        "doi": ids.get("doi"),
        "pmcid": ids.get("pmc"),
        "title": title,
        "abstract": abstract,
        "journal": text_of(art.find("Journal/Title")) or None,
        "year": get_year(art),
        "authors": get_authors(art),
        "mesh_terms": [text_of(d) for d in mc.iterfind("MeshHeadingList/MeshHeading/DescriptorName")],
        "publication_types": pubtypes,
        "language": "eng",
        "source_file": source_file,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }
    return doc, None


# ---------------------------------------------------------------- por arquivo

def parse_file(path_str):
    """Processa um .xml.gz inteiro. Executado em um processo worker."""
    path = Path(path_str)
    out_path = OUT_DIR / path.name.replace(".xml.gz", ".jsonl")
    tmp_path = out_path.with_name(out_path.name + ".part")
    proc = psutil.Process()

    stats = {"file": path.name, "gz_bytes": path.stat().st_size, "records": 0, "kept": 0,
             "dropped": {}, "text_bytes": 0, "out_bytes": 0, "peak_rss_mb": 0.0}
    seen = set()
    t0 = time.perf_counter()

    with gzip.open(path, "rb") as fin, open(tmp_path, "w", encoding="utf-8", newline="\n") as fout:
        context = etree.iterparse(fin, events=("end",), tag="PubmedArticle",
                                  resolve_entities=False, no_network=True)
        for _, pa in context:
            stats["records"] += 1
            doc, reason = parse_record(pa, path.name)
            if doc is not None and doc["pmid"] in seen:
                doc, reason = None, "duplicate"
            if doc is None:
                stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1
            else:
                seen.add(doc["pmid"])
                stats["kept"] += 1
                stats["text_bytes"] += len(doc["title"].encode("utf-8")) + len(doc["abstract"].encode("utf-8"))
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

            # libera memoria: limpa o registro e os irmaos ja processados
            pa.clear(keep_tail=False)
            while pa.getprevious() is not None:
                del pa.getparent()[0]

            if stats["records"] % MEM_SAMPLE_EVERY == 0:
                stats["peak_rss_mb"] = max(stats["peak_rss_mb"], proc.memory_info().rss / 1e6)
        del context

    tmp_path.replace(out_path)
    stats["peak_rss_mb"] = round(max(stats["peak_rss_mb"], proc.memory_info().rss / 1e6), 1)
    stats["out_bytes"] = out_path.stat().st_size
    stats["seconds"] = round(time.perf_counter() - t0, 3)
    stats["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return stats


# ---------------------------------------------------------------- orquestracao

def load_done():
    return set(DONE_FILE.read_text(encoding="utf-8").split()) if DONE_FILE.exists() else set()


def record_done(stats, delete_raw):
    with open(STATS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats) + "\n")
    with open(DONE_FILE, "a", encoding="utf-8") as f:
        f.write(stats["file"] + "\n")
    if delete_raw:
        (RAW_DIR / stats["file"]).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Parser/limpeza do PubMed Baseline")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delete-raw", action="store_true", help="apaga o .gz apos processar com sucesso")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done()
    files = sorted((p for p in RAW_DIR.glob("*.xml.gz") if p.name not in done), reverse=True)
    if not files:
        print("Nenhum arquivo novo em data\\raw.")
        return

    print(f"Processando {len(files)} arquivo(s) com {args.workers} worker(s)...\n")
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(parse_file, str(p)): p.name for p in files}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                s = fut.result()
            except Exception as e:
                print(f"  ERRO em {name}: {e}")
                continue
            record_done(s, args.delete_raw)
            results.append(s)
            print(f"  {s['file']}: {s['records']:,} registros -> {s['kept']:,} mantidos | "
                  f"texto {s['text_bytes'] / 1e6:.1f} MB | {s['seconds']:.1f}s | "
                  f"pico {s['peak_rss_mb']:.0f} MB RAM | descartes {s['dropped']}")

    wall = time.perf_counter() - t0
    gz = sum(s["gz_bytes"] for s in results)
    text = sum(s["text_bytes"] for s in results)
    kept = sum(s["kept"] for s in results)
    print(f"\nResumo: {kept:,} docs | texto puro {text / 1e9:.3f} GB | "
          f"entrada {gz / 1e6:.1f} MB (.gz) | razao texto/gz = {text / max(gz, 1):.2f} | "
          f"tempo {wall:.1f}s")


if __name__ == "__main__":
    main()
