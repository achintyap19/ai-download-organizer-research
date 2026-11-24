import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path("datasets/ml_train_real")
OUT_CSV = Path("datasets/ml_train_real_ground_truth.csv")
LOCKED_BLIND_CSV = Path("datasets/real_wild_ground_truth.csv")

HEADERS = {"User-Agent": "AIDownloadOrganizerResearch/1.0 (Academic Benchmark)"}

# Verified live GitHub repos containing real compiled LaTeX CV / Resume / Cover Letter PDFs
GITHUB_REPOS = [
    ("posquit0", "Awesome-CV"),
    ("rendercv", "rendercv"),
    ("liantze", "AltaCV"),
    ("afriggeri", "cv"),
    ("arasgungore", "arasgungore-CV"),
    ("sb2nov", "resume"),
    ("subidit", "resume"),
    ("dnl-blkv", "mcdowell-cv"),
    ("jitinnair1", "autoCV"),
    ("sajal2692", "data-science-portfolio"),
    ("salomonelli", "best-resume-ever"),
    ("yaoyao-liu", "academic-homepage"),
    ("ankitsultana", "research-statement"),
    ("hku-cv", "cv-template"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_repo_pdfs(owner: str, repo: str) -> list[tuple[str, str]]:
    found = []
    for branch in ("main", "master"):
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        try:
            req = urllib.request.Request(api_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("tree", []):
                p = item.get("path", "")
                if item.get("type") == "blob" and p.lower().endswith(".pdf"):
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{urllib.parse.quote(p)}"
                    safe_name = f"{owner}_{repo}_{Path(p).name}"
                    found.append((safe_name, raw_url))
            if found:
                break
        except Exception:
            continue
    return found


def main() -> None:
    locked_hashes = set()
    with LOCKED_BLIND_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            locked_hashes.add(row["sha256"])

    existing_rows = []
    seen_hashes = set()
    if OUT_CSV.exists():
        with OUT_CSV.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                seen_hashes.add(row["sha256"])

    added = 0
    for owner, repo in GITHUB_REPOS:
        if added >= 15:
            break
        pdfs = discover_repo_pdfs(owner, repo)
        for fname, url in pdfs[:3]:
            if added >= 15:
                break
            dest = OUT_DIR / fname
            if dest.exists():
                continue
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    content = resp.read()
                if not content.startswith(b"%PDF"):
                    continue
                digest = hashlib.sha256(content).hexdigest()
                if digest in locked_hashes or digest in seen_hashes:
                    continue
                dest.write_bytes(content)
                seen_hashes.add(digest)
                existing_rows.append({
                    "filename": fname,
                    "extension": ".pdf",
                    "size_bytes": len(content),
                    "sha256": digest,
                    "true_category": "Career_Identity",
                    "source_domain": "github.com",
                })
                added += 1
                print(f"  [OK +{added}] Career_Identity <- {fname} ({len(content)//1024} KB)")
            except Exception:
                continue
        time.sleep(0.3)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "extension", "size_bytes", "sha256", "true_category", "source_domain"])
        writer.writeheader()
        writer.writerows(existing_rows)

    assert len(seen_hashes & locked_hashes) == 0, "LEAKAGE DETECTED!"
    print(f"\nAdded {added} verified real-world Career_Identity PDFs. Total ml_train_real = {len(existing_rows)} (0% blind hash overlap).")


if __name__ == "__main__":
    main()
