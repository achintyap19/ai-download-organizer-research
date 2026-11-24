import csv
import hashlib
import json
import urllib.request
from pathlib import Path

OUT_DIR = Path("datasets/ml_train_real")
OUT_CSV = Path("datasets/ml_train_real_ground_truth.csv")
LOCKED_BLIND_CSV = Path("datasets/real_wild_ground_truth.csv")

HEADERS = {"User-Agent": "AIDownloadOrganizerResearch/1.0 (Academic Benchmark; Python-urllib)"}

# Curated, non-templated public URLs strictly distinct from real_wild_downloads
PUBLIC_TRAIN_MANIFEST = [
    # 1. Academic_Study (arXiv papers & IETF RFCs distinct from blind set)
    ("Academic_Study", "arxiv.org", "1301.3781.pdf", "https://arxiv.org/pdf/1301.3781.pdf"),
    ("Academic_Study", "arxiv.org", "1406.2661.pdf", "https://arxiv.org/pdf/1406.2661.pdf"),
    ("Academic_Study", "arxiv.org", "1412.6980.pdf", "https://arxiv.org/pdf/1412.6980.pdf"),
    ("Academic_Study", "arxiv.org", "1502.03167.pdf", "https://arxiv.org/pdf/1502.03167.pdf"),
    ("Academic_Study", "arxiv.org", "1505.04597.pdf", "https://arxiv.org/pdf/1505.04597.pdf"),
    ("Academic_Study", "arxiv.org", "1506.02640.pdf", "https://arxiv.org/pdf/1506.02640.pdf"),
    ("Academic_Study", "arxiv.org", "1603.04467.pdf", "https://arxiv.org/pdf/1603.04467.pdf"),
    ("Academic_Study", "arxiv.org", "1609.04747.pdf", "https://arxiv.org/pdf/1609.04747.pdf"),
    ("Academic_Study", "arxiv.org", "1704.04861.pdf", "https://arxiv.org/pdf/1704.04861.pdf"),
    ("Academic_Study", "arxiv.org", "1801.04381.pdf", "https://arxiv.org/pdf/1801.04381.pdf"),
    ("Academic_Study", "arxiv.org", "1905.11946.pdf", "https://arxiv.org/pdf/1905.11946.pdf"),
    ("Academic_Study", "arxiv.org", "1910.01108.pdf", "https://arxiv.org/pdf/1910.01108.pdf"),
    ("Academic_Study", "arxiv.org", "2010.11929.pdf", "https://arxiv.org/pdf/2010.11929.pdf"),
    ("Academic_Study", "arxiv.org", "2106.09685.pdf", "https://arxiv.org/pdf/2106.09685.pdf"),
    ("Academic_Study", "ietf.org", "rfc2616.txt", "https://www.rfc-editor.org/rfc/rfc2616.txt"),
    ("Academic_Study", "ietf.org", "rfc7231.txt", "https://www.rfc-editor.org/rfc/rfc7231.txt"),
    ("Academic_Study", "ietf.org", "rfc8259.txt", "https://www.rfc-editor.org/rfc/rfc8259.txt"),
    ("Academic_Study", "ietf.org", "rfc9110.txt", "https://www.rfc-editor.org/rfc/rfc9110.txt"),

    # 2. Finance_Invoices (US IRS tax forms/schedules & public financial statements distinct from blind set)
    ("Finance_Invoices", "irs.gov", "f1040sa.pdf", "https://www.irs.gov/pub/irs-pdf/f1040sa.pdf"),
    ("Finance_Invoices", "irs.gov", "f1040sb.pdf", "https://www.irs.gov/pub/irs-pdf/f1040sb.pdf"),
    ("Finance_Invoices", "irs.gov", "f1040sc.pdf", "https://www.irs.gov/pub/irs-pdf/f1040sc.pdf"),
    ("Finance_Invoices", "irs.gov", "f1040sd.pdf", "https://www.irs.gov/pub/irs-pdf/f1040sd.pdf"),
    ("Finance_Invoices", "irs.gov", "f1040se.pdf", "https://www.irs.gov/pub/irs-pdf/f1040se.pdf"),
    ("Finance_Invoices", "irs.gov", "f1099int.pdf", "https://www.irs.gov/pub/irs-pdf/f1099int.pdf"),
    ("Finance_Invoices", "irs.gov", "f1099div.pdf", "https://www.irs.gov/pub/irs-pdf/f1099div.pdf"),
    ("Finance_Invoices", "irs.gov", "f1099g.pdf", "https://www.irs.gov/pub/irs-pdf/f1099g.pdf"),
    ("Finance_Invoices", "irs.gov", "f1099k.pdf", "https://www.irs.gov/pub/irs-pdf/f1099k.pdf"),
    ("Finance_Invoices", "irs.gov", "f1099r.pdf", "https://www.irs.gov/pub/irs-pdf/f1099r.pdf"),
    ("Finance_Invoices", "irs.gov", "f941_train.pdf", "https://www.irs.gov/pub/irs-pdf/f941.pdf"),
    ("Finance_Invoices", "irs.gov", "f8889.pdf", "https://www.irs.gov/pub/irs-pdf/f8889.pdf"),
    ("Finance_Invoices", "irs.gov", "f8949.pdf", "https://www.irs.gov/pub/irs-pdf/f8949.pdf"),
    ("Finance_Invoices", "irs.gov", "fw4.pdf", "https://www.irs.gov/pub/irs-pdf/fw4.pdf"),

    # 3. Career_Identity (20 diverse real-world public CVs, resumes, cover letters & career profiles)
    ("Career_Identity", "github.com", "jakes_resume_template.pdf", "https://raw.githubusercontent.com/jakegut/resume/master/resume.pdf"),
    ("Career_Identity", "github.com", "deedy_resume_reversed.pdf", "https://raw.githubusercontent.com/deedy/Deedy-Resume/master/Deedy-Resume-Reversed.pdf"),
    ("Career_Identity", "github.com", "moderncv_classic_en.pdf", "https://raw.githubusercontent.com/xdanaux/moderncv/master/examples/template.pdf"),
    ("Career_Identity", "github.com", "altacv_sample.pdf", "https://raw.githubusercontent.com/liantze/AltaCV/main/sample.pdf"),
    ("Career_Identity", "github.com", "rendercv_classic_sample.pdf", "https://raw.githubusercontent.com/rendercv/rendercv/main/examples/John_Doe_ClassicTheme_CV.pdf"),
    ("Career_Identity", "github.com", "rendercv_sb2nov_sample.pdf", "https://raw.githubusercontent.com/rendercv/rendercv/main/examples/John_Doe_Sb2novTheme_CV.pdf"),
    ("Career_Identity", "github.com", "rendercv_moderncv_sample.pdf", "https://raw.githubusercontent.com/rendercv/rendercv/main/examples/John_Doe_ModerncvTheme_CV.pdf"),
    ("Career_Identity", "github.com", "rendercv_engineering_sample.pdf", "https://raw.githubusercontent.com/rendercv/rendercv/main/examples/John_Doe_EngineeringresumesTheme_CV.pdf"),
    ("Career_Identity", "github.com", "data_scientist_resume_sample.pdf", "https://raw.githubusercontent.com/sajal2692/data-science-portfolio/master/resume/Sajal_Sharma_Resume.pdf"),
    ("Career_Identity", "github.com", "software_engineer_cv_sample.pdf", "https://raw.githubusercontent.com/jitinnair1/autoCV/main/autoCV.pdf"),
    ("Career_Identity", "github.com", "academic_cv_template.pdf", "https://raw.githubusercontent.com/latextemplates/academic-cv/master/cv.pdf"),
    ("Career_Identity", "github.com", "twenty_seconds_cv.pdf", "https://raw.githubusercontent.com/spagnuolocarmine/TwentySecondsCurriculumVitae-LaTex/master/template.pdf"),
    ("Career_Identity", "github.com", "friggeri_cv_sample.pdf", "https://raw.githubusercontent.com/afriggeri/cv/master/cv.pdf"),
    ("Career_Identity", "github.com", "developer_cv_sample.pdf", "https://raw.githubusercontent.com/arasgungore/arasgungore-CV/main/arasgungore-CV.pdf"),
    ("Career_Identity", "github.com", "swe_resume_sample.pdf", "https://raw.githubusercontent.com/sb2nov/resume/master/resume.pdf"),
    ("Career_Identity", "github.com", "ml_engineer_resume.pdf", "https://raw.githubusercontent.com/subidit/resume/master/resume.pdf"),
    ("Career_Identity", "github.com", "research_scientist_cv.pdf", "https://raw.githubusercontent.com/dnl-blkv/mcdowell-cv/master/McDowell_CV_Template.pdf"),
    ("Career_Identity", "github.com", "modern_resume_cover_letter.pdf", "https://raw.githubusercontent.com/posquit0/Awesome-CV/master/examples/coverletter.pdf"),

    # 4. Spreadsheets_Data (Seaborn CSVs distinct from blind set)
    ("Spreadsheets_Data", "seaborn_data", "anscombe.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/anscombe.csv"),
    ("Spreadsheets_Data", "seaborn_data", "attention.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/attention.csv"),
    ("Spreadsheets_Data", "seaborn_data", "brain_networks.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/brain_networks.csv"),
    ("Spreadsheets_Data", "seaborn_data", "car_crashes.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/car_crashes.csv"),
    ("Spreadsheets_Data", "seaborn_data", "dots.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/dots.csv"),
    ("Spreadsheets_Data", "seaborn_data", "dowjones.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/dowjones.csv"),
    ("Spreadsheets_Data", "seaborn_data", "exercise.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/exercise.csv"),
    ("Spreadsheets_Data", "seaborn_data", "flights.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"),
    ("Spreadsheets_Data", "seaborn_data", "fmri.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/fmri.csv"),
    ("Spreadsheets_Data", "seaborn_data", "geyser.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv"),
    ("Spreadsheets_Data", "seaborn_data", "glue.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/glue.csv"),
    ("Spreadsheets_Data", "seaborn_data", "healthexp.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/healthexp.csv"),
    ("Spreadsheets_Data", "seaborn_data", "mpg.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"),
    ("Spreadsheets_Data", "seaborn_data", "planets.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/planets.csv"),
    ("Spreadsheets_Data", "seaborn_data", "seaice.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/seaice.csv"),
    ("Spreadsheets_Data", "seaborn_data", "taxis.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"),
    ("Spreadsheets_Data", "seaborn_data", "tips.csv", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"),

    # 5. Archives_Compressed (GitHub source & release archives distinct from blind set)
    ("Archives_Compressed", "github.com", "flask-3.0.3.tar.gz", "https://github.com/pallets/flask/archive/refs/tags/3.0.3.tar.gz"),
    ("Archives_Compressed", "github.com", "werkzeug-3.0.4.tar.gz", "https://github.com/pallets/werkzeug/archive/refs/tags/3.0.4.tar.gz"),
    ("Archives_Compressed", "github.com", "jinja-3.1.4.tar.gz", "https://github.com/pallets/jinja/archive/refs/tags/3.1.4.tar.gz"),
    ("Archives_Compressed", "github.com", "markupsafe-2.1.5.zip", "https://github.com/pallets/markupsafe/archive/refs/tags/2.1.5.zip"),
    ("Archives_Compressed", "github.com", "itsdangerous-2.2.0.zip", "https://github.com/pallets/itsdangerous/archive/refs/tags/2.2.0.zip"),
    ("Archives_Compressed", "github.com", "tenacity-9.0.0.tar.gz", "https://github.com/jd/tenacity/archive/refs/tags/9.0.0.tar.gz"),
    ("Archives_Compressed", "github.com", "structlog-24.4.0.zip", "https://github.com/hynek/structlog/archive/refs/tags/24.4.0.zip"),
    ("Archives_Compressed", "github.com", "attrs-24.2.0.tar.gz", "https://github.com/python-attrs/attrs/archive/refs/tags/24.2.0.tar.gz"),

    # 6. Media_Photos_Graphics (12 diverse .jpg & .png images distinct from blind set)
    ("Media_Photos_Graphics", "wikimedia.org", "Tux.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Tux.svg/480px-Tux.svg.png"),
    ("Media_Photos_Graphics", "wikimedia.org", "Python-logo.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/480px-Python-logo-notext.svg.png"),
    ("Media_Photos_Graphics", "wikimedia.org", "Earth_Western_Hemisphere.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Earth_Western_Hemisphere_transparent_background.png/480px-Earth_Western_Hemisphere_transparent_background.png"),
    ("Media_Photos_Graphics", "wikimedia.org", "Mona_Lisa.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"),
    ("Media_Photos_Graphics", "wikimedia.org", "Sunflower_sky_backdrop.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/480px-Sunflower_sky_backdrop.jpg"),
    ("Media_Photos_Graphics", "wikimedia.org", "Taj_Mahal_Edited.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Taj_Mahal_%28Edited%29.jpeg/640px-Taj_Mahal_%28Edited%29.jpeg"),
    ("Media_Photos_Graphics", "wikimedia.org", "500px-Golden_Gate_Bridge.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/500px-GoldenGateBridge-001.jpg"),
    ("Media_Photos_Graphics", "wikimedia.org", "500px-Mount_Everest.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/500px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"),
    ("Media_Photos_Graphics", "wikimedia.org", "500px-Eiffel_Tower.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/360px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"),
    ("Media_Photos_Graphics", "wikimedia.org", "500px-Red_Panda.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Ailurus_fulgens_RoterPanda_LesserPanda.jpg/500px-Ailurus_fulgens_RoterPanda_LesserPanda.jpg"),
    ("Media_Photos_Graphics", "github.com", "matplotlib_sample.png", "https://raw.githubusercontent.com/matplotlib/matplotlib/main/doc/_static/logo2.png"),
    ("Media_Photos_Graphics", "github.com", "scikit_learn_logo.png", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo-notext.png"),
]

PYPI_PACKAGES = [
    "flask", "pydantic", "uvicorn", "starlette", "anyio", "certifi",
    "urllib3", "idna", "charset-normalizer", "packaging", "pluggy", "iniconfig",
    "typer", "tqdm", "colorama", "pyyaml",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_pypi_wheel_entries() -> list[tuple[str, str, str, str]]:
    entries = []
    for pkg in PYPI_PACKAGES:
        try:
            url = f"https://pypi.org/pypi/{pkg}/json"
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for file_info in data.get("urls", []):
                if file_info.get("packagetype") == "bdist_wheel" and file_info["filename"].endswith(".whl"):
                    entries.append(("Software_Installers", "pypi.org", file_info["filename"], file_info["url"]))
                    break
        except Exception as exc:
            print(f"  [WARN] PyPI lookup failed for {pkg}: {exc}")
    return entries


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    locked_hashes = set()
    with LOCKED_BLIND_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            locked_hashes.add(row["sha256"])

    manifest = list(PUBLIC_TRAIN_MANIFEST) + fetch_pypi_wheel_entries()
    seen_hashes = set()
    rows = []

    for cat, src_domain, fname, url in manifest:
        dest = OUT_DIR / fname
        if not dest.exists():
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=25) as resp:
                    dest.write_bytes(resp.read())
                print(f"  [OK] {cat:22s} <- {fname}")
            except Exception as exc:
                print(f"  [SKIP] {fname}: {exc}")
                continue

        if not dest.exists() or dest.stat().st_size == 0:
            continue

        digest = sha256_file(dest)
        if digest in locked_hashes:
            print(f"  [LEAKAGE GUARD REMOVED] {fname} matched locked blind hash!")
            dest.unlink(missing_ok=True)
            continue
        if digest in seen_hashes:
            dest.unlink(missing_ok=True)
            continue

        seen_hashes.add(digest)
        rows.append({
            "filename": fname,
            "extension": dest.suffix.lower(),
            "size_bytes": dest.stat().st_size,
            "sha256": digest,
            "true_category": cat,
            "source_domain": src_domain,
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "extension", "size_bytes", "sha256", "true_category", "source_domain"])
        writer.writeheader()
        writer.writerows(rows)

    overlap = len(seen_hashes & locked_hashes)
    assert overlap == 0, f"CRITICAL DATA LEAKAGE: {overlap} overlapping hashes!"
    print(f"\nSaved {len(rows)} unique real-world training files to {OUT_CSV} (SHA-256 overlap with blind test: {overlap}).")


if __name__ == "__main__":
    main()
