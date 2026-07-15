"""
_bootstrap.py — robust setup for exp5/exp6 on Kaggle / Colab / local.

Solves two recurring Kaggle problems:
  1. `pip install git+https://github.com/ARYAN2302/tiny-cl.git` fails with
     exit 128 (auth, rate-limit, or network). Fix: try pip-git -> git clone
     -> raw-file inline, in that order.
  2. `modelscope.snapshot_download(...)` fails (SDK hiccups, network, or
     model not mirrored). Fix: try ModelScope -> HF mirror (hf-mirror.com
     with xet disabled) -> direct HuggingFace, in that order.

Usage from an exp script:

    import os, sys
    try:
        from _bootstrap import (install_deps, install_avr, download_model,
                                patch_transformers_torch26, OUTPUT_DIR, DATA_CACHE)
    except ImportError:
        import urllib.request, tempfile
        _src = "https://raw.githubusercontent.com/ARYAN2302/tiny-cl/main/scripts/_bootstrap.py"
        _dst = os.path.join(tempfile.gettempdir(), "_bootstrap.py")
        urllib.request.urlretrieve(_src, _dst)
        sys.path.insert(0, tempfile.gettempdir())
        from _bootstrap import (install_deps, install_avr, download_model,
                                patch_transformers_torch26, OUTPUT_DIR, DATA_CACHE)

    install_deps()                  # peft, accelerate, modelscope, huggingface_hub, ...
    install_avr()                   # installs the `avr` package (3 fallback strategies)
    patch_transformers_torch26()    # work around torch>=2.6 + older transformers check
    MODEL_PATH = download_model("Qwen/Qwen3-1.7B", cache_dir=OUTPUT_DIR / "model_cache")
"""
import os
import sys
import re
import subprocess
import urllib.request
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Env tweaks — must be set before any HF/torch import.
# HF_HUB_DISABLE_XET=1 is the key fix for the hf-xet 403 errors that forced
# the switch to ModelScope in the first place. With xet disabled, regular
# HTTPS downloads from HF work fine from Kaggle.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Output dirs — Kaggle /kaggle/working if present, else ./output
OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE = OUTPUT_DIR / "data_cache"
DATA_CACHE.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/ARYAN2302/tiny-cl.git"
REPO_RAW = "https://raw.githubusercontent.com/ARYAN2302/tiny-cl/main"


# ---------------------------------------------------------------------------
# 1. Dependency install
# ---------------------------------------------------------------------------
def _pip(*args, check=True, timeout=600, quiet=True):
    """Run pip with a timeout. On failure, print captured stderr before raising."""
    flags = ["-q"] if quiet else []
    cmd = [sys.executable, "-m", "pip", "install", *flags, *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and check:
        # Surface the actual pip error — the -q flag hides it by default
        print(f"[bootstrap] pip FAILED: {' '.join(args)}", flush=True)
        print(f"[bootstrap] pip returncode: {r.returncode}", flush=True)
        if r.stderr:
            # Print last 2000 chars of stderr (the useful part)
            print("[bootstrap] pip stderr (tail):\n" + r.stderr[-2000:], flush=True)
        if r.stdout:
            print("[bootstrap] pip stdout (tail):\n" + r.stdout[-1000:], flush=True)
        raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    return r


def _pip_maybe(*args, timeout=600):
    """Run pip, return True on success, False on failure (no raise)."""
    try:
        r = _pip(*args, check=False, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[bootstrap] pip timed out: {' '.join(args)}", flush=True)
        return False


def install_deps(extra=None, transformers_pin=None):
    """
    Install common deps for exp5/exp6. Resilient to individual package failures.

    Strategy:
      1. Try installing all packages in one batch (fast path).
      2. If that fails, install them one at a time so a single bad package
         doesn't kill the whole run. Print a warning for each failure.
      3. modelscope is OPTIONAL — if it fails to install, we skip it and
         rely on the HF fallback in download_model(). Same for transformers_pin.

    Args:
        extra: list of extra pip specs to install.
        transformers_pin: e.g. ">=4.57.0,<5.0.0" for LFM2.5 support.
                         If None, installs whatever is already there (no force).
    """
    # Core packages — these are required and must all install.
    core_pkgs = [
        "peft>=0.13.0",
        "accelerate>=1.0.0",
        "sentencepiece",
        "protobuf",
        "packaging",
        "huggingface_hub>=0.26.0",  # needed for HF_HUB_DISABLE_XET support
        "modelscope",               # REQUIRED — only reliable model source on Kaggle
    ]
    # Optional packages — if they fail, we continue
    optional_pkgs = []
    if transformers_pin:
        optional_pkgs.append(f"transformers{transformers_pin}")
    if extra:
        optional_pkgs.extend(extra)

    print("[bootstrap] Installing deps...", flush=True)

    # Try batch install of core + optional first (fast path)
    all_pkgs = core_pkgs + optional_pkgs
    if _pip_maybe(*all_pkgs):
        print("[bootstrap] All deps installed in one batch.", flush=True)
    else:
        # Batch failed — install core one-by-one (required), then optional (best-effort)
        print("[bootstrap] Batch install failed, installing individually...", flush=True)
        for pkg in core_pkgs:
            if not _pip_maybe(pkg):
                # Core package failed — this is fatal, but try without version pin
                pkg_name = re.split(r"[><=!]", pkg, maxsplit=1)[0]
                print(f"[bootstrap] WARNING: {pkg} failed, trying {pkg_name} unpinned...", flush=True)
                if not _pip_maybe(pkg_name):
                    raise RuntimeError(f"Required package {pkg_name} failed to install")
        for pkg in optional_pkgs:
            if not _pip_maybe(pkg):
                pkg_name = re.split(r"[><=!]", pkg, maxsplit=1)[0]
                print(f"[bootstrap] WARNING: optional {pkg} failed ({pkg_name} unavailable). "
                      f"Will use fallbacks.", flush=True)

    # torchao crashes on T4; hf-xet is what causes the 403s. Kill both.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False, capture_output=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "hf-xet"],
                   check=False, capture_output=True)
    print("[bootstrap] Deps ready.", flush=True)


# ---------------------------------------------------------------------------
# 2. avr package install — three fallback strategies
# ---------------------------------------------------------------------------
def _verify_avr_installed():
    """
    Light sanity check: verify the avr package is importable *as a module*
    (i.e., its files are findable on sys.path). We do NOT do a full
    `import avr` here because that would trigger `import torch` etc.,
    which may not be installed yet at this point in the bootstrap.

    Instead, we use importlib.util.find_spec to check that the package
    metadata is resolvable without executing the module body.
    """
    import importlib
    importlib.invalidate_caches()
    # find_spec resolves the package location without importing it.
    # It will raise ModuleNotFoundError only if `avr` itself is not on the path.
    spec = importlib.util.find_spec("avr")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("avr package not found on sys.path")
    return spec.origin


def install_avr(repo_url=REPO_URL, local_dir="/tmp/tiny-cl-src"):
    """
    Install the `avr` package. Returns the method that worked.

    Strategy 1: pip install git+<repo_url>            (fast, needs github access)
    Strategy 2: git clone + pip install -e <local>    (also needs github)
    Strategy 3: download raw .py files from github raw (works even if git
                protocol is blocked, as long as HTTPS to raw.githubusercontent.com works)
    """
    # Strategy 1
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             f"git+{repo_url}"],
            capture_output=True, text=True, timeout=240)
        if r.returncode == 0:
            try:
                _verify_avr_installed()
                print(f"[bootstrap] avr installed via pip-git", flush=True)
                return "pip-git"
            except ModuleNotFoundError:
                print(f"[bootstrap] pip-git rc=0 but avr not findable; trying next", flush=True)
        else:
            print(f"[bootstrap] pip-git failed (rc={r.returncode}): "
                  f"{r.stderr.strip()[-200:]}", flush=True)
    except subprocess.TimeoutExpired:
        print("[bootstrap] pip-git timed out", flush=True)
    except Exception as e:
        print(f"[bootstrap] pip-git exception: {e}", flush=True)

    # Strategy 2
    try:
        if not Path(local_dir).exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, local_dir],
                check=True, capture_output=True, text=True, timeout=240)
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "-e", local_dir],
            capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            # For editable installs, also add to sys.path as fallback
            if str(Path(local_dir)) not in sys.path:
                sys.path.insert(0, str(Path(local_dir)))
            try:
                _verify_avr_installed()
                print(f"[bootstrap] avr installed via git-clone + pip-local", flush=True)
                return "pip-local"
            except ModuleNotFoundError:
                print(f"[bootstrap] pip-local rc=0 but avr not findable; trying next", flush=True)
        else:
            print(f"[bootstrap] pip-local failed (rc={r.returncode}): "
                  f"{r.stderr.strip()[-200:]}", flush=True)
    except subprocess.TimeoutExpired:
        print("[bootstrap] git clone timed out", flush=True)
    except Exception as e:
        print(f"[bootstrap] pip-local exception: {e}", flush=True)

    # Strategy 3 — inline from raw.githubusercontent.com
    return _inline_avr_from_raw(local_dir)


def _inline_avr_from_raw(local_dir):
    """Download avr/*.py from GitHub raw and install as a local package."""
    print("[bootstrap] Falling back to raw-file inline install...", flush=True)
    files = ["__init__.py", "model.py", "learn.py", "verify.py",
             "repair.py", "eval.py", "run.py", "cli.py"]
    pkg_dir = Path(local_dir) / "avr"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    for fn in files:
        url = f"{REPO_RAW}/avr/{fn}"
        dest = pkg_dir / fn
        req = urllib.request.Request(url, headers={"User-Agent": "avr-bootstrap/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"  fetched {fn} ({dest.stat().st_size} bytes)", flush=True)

    # Minimal pyproject.toml so setuptools picks up the package
    (Path(local_dir) / "pyproject.toml").write_text(
        '[build-system]\n'
        'requires = ["setuptools>=68.0", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\n'
        'name = "avr-cl"\n'
        'version = "0.1.0"\n'
        'dependencies = []\n\n'
        '[tool.setuptools.packages.find]\n'
        'include = ["avr*"]\n'
    )

    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "-e", str(Path(local_dir))],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(
            f"All avr install strategies failed.\n"
            f"Last error (pip-local-editable from raw):\n{r.stderr[-1500:]}")
    # Belt-and-suspenders: add the local dir to sys.path so `import avr`
    # works even if the editable install's .pth file wasn't picked up.
    if str(Path(local_dir)) not in sys.path:
        sys.path.insert(0, str(Path(local_dir)))
    _verify_avr_installed()
    print(f"[bootstrap] avr installed via raw-inline", flush=True)
    return "raw-inline"


# ---------------------------------------------------------------------------
# 3. Model download — three fallback sources
# ---------------------------------------------------------------------------
def download_model(model_id, cache_dir, prefer="modelscope"):
    """
    Download a model snapshot. Returns the local path.

    Tries in order:
      1. ModelScope snapshot_download  (fast in China; sometimes flaky on Kaggle)
      2. HuggingFace via hf-mirror.com with xet disabled  (HF_HUB_DISABLE_XET=1)
      3. Direct HuggingFace with xet disabled  (works from US/West Kaggle IPs)

    Args:
        model_id: e.g. "Qwen/Qwen3-1.7B" or "liquidai/LFM2.5-1.2B-Instruct"
        cache_dir: where to cache. Will be created if missing.
        prefer: 'modelscope' (try MS first) or 'hf' (skip MS).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    errors = []

    # --- Source 1: ModelScope (optional — may not be installed) ---
    if prefer != "hf":
        try:
            from modelscope import snapshot_download as ms_download
            path = ms_download(model_id, cache_dir=str(cache_dir))
            print(f"[download] ModelScope OK: {path}", flush=True)
            return path
        except ImportError:
            errors.append("modelscope: not installed (optional, skipped)")
            print(f"[download] ModelScope: not installed, skipping to HF", flush=True)
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            errors.append(f"modelscope: {msg}")
            print(f"[download] ModelScope failed: {msg}", flush=True)

    # --- Source 2: HuggingFace mirror (hf-mirror.com) with xet disabled ---
    # HF_HUB_DISABLE_XET=1 is already set at module load. Reassert + endpoint.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        from huggingface_hub import snapshot_download as hf_download
        path = hf_download(model_id, cache_dir=str(cache_dir))
        print(f"[download] HF mirror (hf-mirror.com) OK: {path}", flush=True)
        return path
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        errors.append(f"hf-mirror: {msg}")
        print(f"[download] HF mirror failed: {msg}", flush=True)

    # --- Source 3: Direct HuggingFace (no mirror) with xet disabled ---
    os.environ.pop("HF_ENDPOINT", None)  # let it resolve to default
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import snapshot_download as hf_download
        path = hf_download(model_id, cache_dir=str(cache_dir))
        print(f"[download] Direct HF OK: {path}", flush=True)
        return path
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        errors.append(f"hf-direct: {msg}")
        print(f"[download] Direct HF failed: {msg}", flush=True)

    raise RuntimeError(
        f"All model download strategies failed for {model_id}:\n" +
        "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# 4. transformers/torch 2.6 workaround
# ---------------------------------------------------------------------------
def patch_transformers_torch26():
    """
    Kaggle ships torch>=2.6 but some transformers versions gate features
    behind is_torch_greater_or_equal_than_2_6 in a way that breaks on T4.
    Force the check to False so transformers uses the 2.5 code path.
    """
    try:
        import transformers.utils.import_utils as _iu
        _iu._is_torch_greater_or_equal_than_2_6 = False
        _iu.is_torch_greater_or_equal_than_2_6 = lambda: False
        print("[bootstrap] Patched transformers torch-2.6 check -> False", flush=True)
    except (AttributeError, ImportError):
        pass
