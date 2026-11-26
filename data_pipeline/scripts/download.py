"""
download.py - Passo 2: baixa arquivos do PubMed Baseline com verificacao MD5,
retentativas automaticas e log de tempo. Ordem: do mais recente para o mais antigo.

Uso (a partir de C:\\pubmed-rag, com o venv ativo):
    python scripts\\download.py --n 2               # baixa os 2 mais recentes (teste)
    python scripts\\download.py --start 1320 --n 5  # baixa 1320, 1319, ..., 1316
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PREFIX = "pubmed26n"
LAST_FILE = 1334

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"
TIMING_LOG = LOG_DIR / "download_times.jsonl"
CHUNK = 1 << 20  # 1 MB


def file_name(i: int) -> str:
    return f"{PREFIX}{i:04d}.xml.gz"


def _open(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "unb-rag-project/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch(url: str, dest: Path, retries: int = 5) -> None:
    """Baixa url para dest via arquivo .part (nunca deixa arquivo pela metade com o nome final)."""
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with _open(url) as r, open(tmp, "wb") as f:
                while chunk := r.read(CHUNK):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [tentativa {attempt}/{retries}] erro: {e} - aguardando {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Falha ao baixar {url} apos {retries} tentativas")


def expected_md5(name: str) -> str:
    """Le o .md5 oficial do NCBI (formato 'MD5(arquivo)= <hash>')."""
    with _open(BASE_URL + name + ".md5") as r:
        text = r.read().decode("ascii", errors="ignore")
    m = re.search(r"\b[0-9a-f]{32}\b", text.lower())
    if not m:
        raise ValueError(f"Nao encontrei hash MD5 em {name}.md5: {text!r}")
    return m.group(0)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def log_timing(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(TIMING_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def download_file(name: str, max_md5_retries: int = 3) -> Path:
    """Baixa e valida um arquivo do baseline. Retorna o caminho local.
    Se o arquivo ja existir e o MD5 bater, nao baixa de novo."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / name
    exp = expected_md5(name)

    if dest.exists() and md5sum(dest) == exp:
        print(f"  {name}: ja existe e MD5 confere - pulando")
        return dest

    for attempt in range(1, max_md5_retries + 1):
        t0 = time.perf_counter()
        fetch(BASE_URL + name, dest)
        elapsed = time.perf_counter() - t0
        size = dest.stat().st_size
        if md5sum(dest) == exp:
            mb = size / 1e6
            print(f"  {name}: {mb:.1f} MB em {elapsed:.1f}s ({mb / elapsed:.1f} MB/s) - MD5 ok")
            log_timing({"file": name, "bytes": size, "seconds": round(elapsed, 3),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
            return dest
        print(f"  {name}: MD5 nao confere (tentativa {attempt}/{max_md5_retries}) - baixando de novo")
        dest.unlink(missing_ok=True)

    raise RuntimeError(f"{name}: MD5 invalido apos {max_md5_retries} tentativas")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download do PubMed Baseline")
    ap.add_argument("--start", type=int, default=LAST_FILE, help="numero do primeiro arquivo (desce a partir dele)")
    ap.add_argument("--n", type=int, default=2, help="quantidade de arquivos")
    args = ap.parse_args()

    names = [file_name(i) for i in range(args.start, max(args.start - args.n, 0), -1)]
    print(f"Baixando {len(names)} arquivo(s) para {RAW_DIR}")

    t0 = time.perf_counter()
    total = 0
    for name in names:
        total += download_file(name).stat().st_size
    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {total / 1e6:.1f} MB em {elapsed:.1f}s ({total / 1e6 / max(elapsed, 1e-9):.1f} MB/s)")


if __name__ == "__main__":
    main()