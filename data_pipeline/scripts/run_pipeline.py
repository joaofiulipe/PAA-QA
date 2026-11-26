"""
run_pipeline.py - Passo 4: baixa (em paralelo) -> valida MD5 -> parseia (em paralelo)
-> apaga o .gz, do arquivo mais recente para o mais antigo, ate atingir a meta de texto puro.

Uso (a partir de C:\\pubmed-rag, com o venv ativo):
    python scripts\\run_pipeline.py                      # meta padrao: 10.5 GB
    python scripts\\run_pipeline.py --target-gb 10.5 --dl-workers 4 --parse-workers 3

Retomavel: se for interrompido, rode de novo; arquivos ja processados (logs\\processed.txt)
sao pulados. A meta pode ser ultrapassada levemente pelos arquivos que ja estavam em
andamento quando ela foi atingida.
"""
import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

from download import LAST_FILE, file_name, download_file
from parse_pubmed import LOG_DIR, OUT_DIR, STATS_LOG, load_done, parse_file, record_done

SUMMARY = LOG_DIR / "pipeline_summary.json"


def previous_stats():
    """Estatisticas dos arquivos ja processados em execucoes anteriores."""
    if not STATS_LOG.exists():
        return []
    with open(STATS_LOG, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Pipeline PubMed: download + parsing ate a meta")
    ap.add_argument("--target-gb", type=float, default=10.5, help="meta de texto puro (titulo+resumo) em GB")
    ap.add_argument("--dl-workers", type=int, default=4, help="downloads simultaneos")
    ap.add_argument("--parse-workers", type=int, default=3, help="processos de parsing")
    ap.add_argument("--max-raw", type=int, default=8, help="max. de .gz em disco/em andamento")
    ap.add_argument("--start", type=int, default=LAST_FILE, help="arquivo inicial (desce a partir dele)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    target = args.target_gb * 1e9
    done = load_done()
    initial_text = sum(s["text_bytes"] for s in previous_stats())
    total_text = initial_text
    candidates = iter([file_name(i) for i in range(args.start, 0, -1) if file_name(i) not in done])

    print(f"Meta: {args.target_gb} GB | ja processado: {initial_text / 1e9:.3f} GB "
          f"({len(done)} arquivos) | downloads: {args.dl_workers} | parsers: {args.parse_workers}\n")

    downloads, parses = {}, {}
    new_stats, failed = [], []
    stop = total_text >= target
    t0 = time.perf_counter()

    dl_pool = ThreadPoolExecutor(max_workers=args.dl_workers)
    parse_pool = ProcessPoolExecutor(max_workers=args.parse_workers)
    try:
        while True:
            # agenda novos downloads enquanto ha espaco e a meta nao foi atingida
            while (not stop and len(downloads) < args.dl_workers
                   and len(downloads) + len(parses) < args.max_raw):
                name = next(candidates, None)
                if name is None:
                    stop = True
                    break
                downloads[dl_pool.submit(download_file, name)] = name

            if not downloads and not parses:
                break

            finished, _ = wait(list(downloads) + list(parses), return_when=FIRST_COMPLETED)
            for fut in finished:
                if fut in downloads:
                    name = downloads.pop(fut)
                    try:
                        path = fut.result()
                    except Exception as e:
                        print(f"  ERRO no download de {name}: {e}")
                        failed.append(name)
                        continue
                    parses[parse_pool.submit(parse_file, str(path))] = name
                else:
                    name = parses.pop(fut)
                    try:
                        s = fut.result()
                    except Exception as e:
                        print(f"  ERRO no parsing de {name}: {e}")
                        failed.append(name)
                        continue
                    record_done(s, delete_raw=True)
                    new_stats.append(s)
                    total_text += s["text_bytes"]

                    elapsed = time.perf_counter() - t0
                    rate = (total_text - initial_text) / max(elapsed, 1e-9)
                    eta = (target - total_text) / rate if rate > 0 and total_text < target else 0
                    print(f"[{len(new_stats):>3}] {name}: +{s['text_bytes'] / 1e6:.1f} MB | "
                          f"total {total_text / 1e9:.2f}/{args.target_gb} GB "
                          f"({100 * total_text / target:.1f}%) | {fmt_time(elapsed)} | ETA {fmt_time(eta)}")

                    if total_text >= target and not stop:
                        stop = True
                        print("\nMeta atingida - finalizando arquivos em andamento...\n")
    except KeyboardInterrupt:
        print("\nInterrompido. Rode o mesmo comando de novo para continuar de onde parou.")
        dl_pool.shutdown(wait=False, cancel_futures=True)
        parse_pool.shutdown(wait=False, cancel_futures=True)
        return
    dl_pool.shutdown()
    parse_pool.shutdown()

    wall = time.perf_counter() - t0
    all_stats = previous_stats()
    summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_gb": args.target_gb,
        "files_processed_total": len(all_stats),
        "files_this_run": len(new_stats),
        "failed_files": failed,
        "documents_total": sum(s["kept"] for s in all_stats),
        "records_read_total": sum(s["records"] for s in all_stats),
        "text_bytes_total": sum(s["text_bytes"] for s in all_stats),
        "jsonl_bytes_total": sum(s["out_bytes"] for s in all_stats),
        "gz_bytes_total": sum(s["gz_bytes"] for s in all_stats),
        "wall_seconds_this_run": round(wall, 1),
        "parse_cpu_seconds_total": round(sum(s["seconds"] for s in all_stats), 1),
        "dl_workers": args.dl_workers,
        "parse_workers": args.parse_workers,
    }
    dropped = {}
    for s in all_stats:
        for k, v in s["dropped"].items():
            dropped[k] = dropped.get(k, 0) + v
    summary["dropped_total"] = dropped
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Concluido: {summary['documents_total']:,} docs | "
          f"texto puro {summary['text_bytes_total'] / 1e9:.2f} GB | "
          f"JSONL {summary['jsonl_bytes_total'] / 1e9:.2f} GB | tempo {fmt_time(wall)}")
    if failed:
        print(f"Falharam (rode de novo para tentar): {failed}")
    print(f"Resumo salvo em {SUMMARY}")


if __name__ == "__main__":
    main()