#!/usr/bin/env python3
"""Checks for the pip install profiles (pyproject_<profile>.toml).

lint   Static checks. With --online, also checks that every pin is downloadable for the hosts and that every
       git commit can be fetched.
image  Runs inside the profile's reference image: pins and git commits match the image, the pinned closure has
       no conflicts beyond the --no-deps overrides, and no unpinned package in it would drift on PyPI. Writes a
       snapshot of the image for `venv`.
venv   Runs with the venv's python after installing: the venv matches the profile and the image snapshot.
"""

import argparse
import importlib
import importlib.metadata as metadata
import json
import pkgutil
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.dependency_groups import DependencyGroupResolver
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPI = "https://pypi.org/pypi"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
GIT_URL = re.compile(r"^git\+(?P<repo>https://[^@\s]+)@(?P<sha>[0-9a-f]{40})(?:#subdirectory=[\w./-]+)?$")
# setup.py metadata that every profile overrides on purpose.
EXPECTED_SETUPTOOLS_WARNINGS = (
    "`license` overwritten by `pyproject.toml`",
    "`extras_require` overwritten in `pyproject.toml`",
)
# Guarded optional imports that no profile installs.
OPTIONAL_IMPORTS = {"gemlite", "torchao", "triton_kernels", "vllm", "xielu"}
# The hosts run RHEL 9 (glibc 2.34) and Python 3.12.
HOST_PLATFORMS = [f"manylinux_2_{minor}_x86_64" for minor in range(34, 16, -1)] + [
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
]
HOST_TAGS = set(cpython_tags((3, 12), platforms=HOST_PLATFORMS)) | set(
    compatible_tags((3, 12), "cp312", HOST_PLATFORMS)
)
# Image recipes in this repo whose ARGs must match the profile: profile -> (recipe, {ARG: package}).
RECIPE_ARGS = {
    "sm120": (
        "docker/Dockerfile.flux-sm120",
        {
            "FLASHINFER_COMMIT": "flashinfer-python",
            "FLASH_ATTN_COMMIT": "flash-attn-4",
            "TORCH_VERSION": "torch",
            "TORCHVISION_VERSION": "torchvision",
            "TORCHAUDIO_VERSION": "torchaudio",
            "TRANSFORMERS_VERSION": "transformers",
            "CUTLASS_DSL_VERSION": "nvidia-cutlass-dsl",
        },
    ),
}
SETUPTOOLS_SNIPPET = r"""
import json
import warnings

import setuptools  # noqa: F401  (installs the distutils shim)
import distutils.core

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    dist = distutils.core.run_setup("setup.py", stop_after="config")
print(json.dumps({
    "warnings": [str(w.message).strip().splitlines()[0] for w in caught],
    "version": dist.metadata.version,
    "console_scripts": [str(e) for e in (dist.entry_points or {}).get("console_scripts", [])],
    "extras": sorted(dist.extras_require or {}),
}))
"""


class Profile:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem.removeprefix("pyproject_")
        self.data = tomllib.loads(path.read_text())
        self.project = self.data.get("project", {})
        self.extras: Dict[str, List[str]] = self.project.get("optional-dependencies", {})
        self.groups: Dict[str, List[Any]] = self.data.get("dependency-groups", {})

    def group_requirements(self) -> Dict[str, List[Requirement]]:
        resolver = DependencyGroupResolver(self.groups)
        return {group: list(resolver.resolve(group)) for group in self.groups}

    def requirements(self) -> List[Tuple[str, Requirement]]:
        found = [(f"extra {extra}", Requirement(raw)) for extra, reqs in self.extras.items() for raw in reqs]
        for group, reqs in self.group_requirements().items():
            found.extend((f"group {group}", req) for req in reqs)
        return found

    def pins(self) -> Dict[str, str]:
        return {
            canonicalize_name(req.name): version
            for _, req in self.requirements()
            if (version := exact_version(req)) is not None
        }

    def git_refs(self) -> Dict[str, Tuple[str, str]]:
        refs = {}
        for _, req in self.requirements():
            match = GIT_URL.match(req.url or "")
            if match:
                refs[canonicalize_name(req.name)] = (match.group("repo"), match.group("sha"))
        return refs

    def nodeps_names(self) -> Set[str]:
        return {canonicalize_name(req.name) for reqs in self.group_requirements().values() for req in reqs}

    def roots(self) -> Dict[str, Set[str]]:
        """Packages the extras install, with their requested extras (the groups install with --no-deps)."""
        roots: Dict[str, Set[str]] = {}
        for reqs in self.extras.values():
            for raw in reqs:
                req = Requirement(raw)
                roots.setdefault(canonicalize_name(req.name), set()).update(req.extras)
        return roots


def exact_version(req: Requirement) -> Optional[str]:
    specs = list(req.specifier)
    if len(specs) == 1 and specs[0].operator == "==" and "*" not in specs[0].version:
        return specs[0].version
    return None


def requirement_problem(req: Requirement) -> Optional[str]:
    if req.marker is not None:
        return f"{req}: environment markers are not allowed"
    if req.url:
        if not GIT_URL.match(req.url):
            return f"{req}: a URL requirement must be git+https://...@<40-hex commit>"
        return None
    if exact_version(req) is None:
        return f"{req}: must be an exact == pin"
    return None


def duplicate_names(reqs: Iterable[Requirement]) -> List[str]:
    seen: Set[str] = set()
    duplicates = []
    for req in reqs:
        name = canonicalize_name(req.name)
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def parse_recipe_args(text: str) -> Dict[str, str]:
    return dict(re.findall(r"^ARG (\w+)=(\S+)", text, flags=re.MULTILINE))


def first_wins(entries: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    """Keep the first entry per canonical name, like imports and pip do along sys.path."""
    kept: Dict[str, Any] = {}
    for name, value in entries:
        kept.setdefault(canonicalize_name(name), value)
    return kept


def installed_dists() -> Dict[str, metadata.Distribution]:
    return first_wins((dist.metadata["Name"], dist) for dist in metadata.distributions())


def dependency_conflicts(requires: Dict[str, List[str]], versions: Dict[str, str]) -> Set[Tuple[str, str]]:
    """(dist, dependency) pairs that `pip check` reports: a base requirement missing or at the wrong version."""
    conflicts = set()
    for dist, reqs in requires.items():
        for raw in reqs:
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            dep = canonicalize_name(req.name)
            if dep not in versions or not req.specifier.contains(versions[dep], prereleases=True):
                conflicts.add((dist, dep))
    return conflicts


def commit_matches(pinned: str, found: Optional[str]) -> bool:
    """`found` may be a full or an abbreviated (>= 7 hex digits) commit."""
    return bool(found) and len(found) >= 7 and pinned.startswith(found)


def public_version(text: str) -> Version:
    """`2.11.0+cu130` -> 2.11.0; equal releases compare equal (5.17 == 5.17.0)."""
    return Version(Version(text).public)


def cuda_from_local(version: str) -> Optional[str]:
    """`2.11.0+cu130` -> `13.0`."""
    match = re.fullmatch(r"cu(\d+)(\d)", Version(version).local or "")
    return f"{match.group(1)}.{match.group(2)}" if match else None


def _open(url: str, attempts: int = 3) -> Optional[bytes]:
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "fluxserve-ci"}), timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2 * (attempt + 1))
    return None


def _host_wheel(filename: str) -> bool:
    try:
        return bool(set(parse_wheel_filename(filename)[3]) & HOST_TAGS)
    except InvalidWheelFilename:
        return False


def availability_problem(name: str, version: str) -> Optional[str]:
    if "+" in version:
        page = (_open(f"{TORCH_INDEX}/{name}/") or b"").decode()
        wheels = set(re.findall(r">([^<>]+\.whl)<", page))
        if not any(
            _host_wheel(wheel) and parse_wheel_filename(wheel)[1] == Version(version)
            for wheel in wheels
            if wheel.endswith(".whl")
        ):
            return f"{name}=={version}: no cp312 manylinux_2_34 wheel on {TORCH_INDEX}"
        return None
    raw = _open(f"{PYPI}/{name}/{version}/json")
    if raw is None:
        return f"{name}=={version}: not on PyPI"
    files = [f for f in json.loads(raw)["urls"] if not f.get("yanked")]
    if not files:
        return f"{name}=={version}: every file is yanked"
    if not any(_host_wheel(f["filename"]) for f in files if f["filename"].endswith(".whl")):
        return f"{name}=={version}: no wheel for cp312 manylinux_2_34 (would build from source)"
    return None


def fetch_problem(repo: str, sha: str) -> Optional[str]:
    """pip clones the repository and checks the commit out; fail if the commit can't be fetched."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        result = subprocess.run(
            ["git", "-C", tmp, "fetch", "-q", "--depth=1", "--filter=blob:none", repo, sha],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        return f"{repo}@{sha}: cannot fetch the commit ({result.stderr.strip()})"
    return None


def setuptools_problems(profile: Profile) -> List[str]:
    """Run setup.py's config stage with the profile as pyproject.toml, in a scratch copy of the tree."""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(REPO_ROOT / "setup.py", tmp)
        shutil.copy(profile.path, Path(tmp) / "pyproject.toml")
        (Path(tmp) / "python").symlink_to(REPO_ROOT / "python")
        result = subprocess.run([sys.executable, "-c", SETUPTOOLS_SNIPPET], cwd=tmp, capture_output=True, text=True)
    if result.returncode != 0:
        return [f"setup.py config stage failed:\n{result.stderr.strip()}"]
    info = json.loads(result.stdout.strip().splitlines()[-1])
    problems = [
        f"setuptools warning: {message}"
        for message in info["warnings"]
        if "pyproject.toml" in message and not message.startswith(EXPECTED_SETUPTOOLS_WARNINGS)
    ]
    init = (REPO_ROOT / "python" / "fluxserve" / "__init__.py").read_text()
    init_version = re.search(r"^__version__ = ['\"]([^'\"]+)['\"]", init, flags=re.MULTILINE).group(1)
    if not info["version"].startswith(init_version):
        problems.append(f"version {info['version']} does not come from setup.py ({init_version})")
    if "fluxserve = fluxserve.cli.app:main" not in info["console_scripts"]:
        problems.append(f"missing the fluxserve console script: {info['console_scripts']}")
    if set(info["extras"]) != {profile.name, "dev"}:
        problems.append(f"built extras {info['extras']} are not [{profile.name}, dev]")
    return problems


def lint_problems(profile: Profile, online: bool) -> List[str]:
    problems = []
    if profile.project.get("dependencies") != []:
        problems.append("[project] dependencies must stay empty; put pins in the extras")
    missing_dynamic = {"version", "requires-python"} - set(profile.project.get("dynamic", []))
    if missing_dynamic:
        problems.append(f"[project] dynamic must include {sorted(missing_dynamic)} (setup.py sets them)")
    if set(profile.extras) != {profile.name, "dev"}:
        problems.append(f"extras {sorted(profile.extras)} must be exactly [{profile.name}, dev]")
    try:
        reqs = profile.requirements()
    except Exception as exc:  # malformed requirement or dependency group
        return problems + [f"cannot parse requirements: {exc}"]
    problems += [problem for _, req in reqs if (problem := requirement_problem(req))]
    problems += [f"{name} is listed more than once" for name in duplicate_names(req for _, req in reqs)]
    problems += setuptools_problems(profile)
    if profile.name in RECIPE_ARGS:
        recipe, arg_packages = RECIPE_ARGS[profile.name]
        args = parse_recipe_args((REPO_ROOT / recipe).read_text())
        pins, refs = profile.pins(), profile.git_refs()
        for arg, package in arg_packages.items():
            value = args.get(arg)
            if package in refs:
                expected = refs[package][1]
                ok = value == expected
            else:
                expected = pins.get(package, "<unpinned>")
                ok = value is not None and package in pins and public_version(value) == public_version(expected)
            if not ok:
                problems.append(f"{recipe} {arg}={value} does not match the profile's {package} ({expected})")
    if online:
        with ThreadPoolExecutor(16) as pool:
            problems += [p for p in pool.map(lambda item: availability_problem(*item), profile.pins().items()) if p]
            problems += [p for p in pool.map(lambda ref: fetch_problem(*ref), profile.git_refs().values()) if p]
    return problems


def installed_commit(dist: metadata.Distribution) -> Optional[str]:
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    commit = info.get("vcs_info", {}).get("commit_id")
    if commit:
        return commit
    url = info.get("url", "")
    if url.startswith("file://"):
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", url[len("file://"):], "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    for path in dist.files or []:  # FlashInfer records its commit at build time.
        if path.name == "_build_meta.py":
            match = re.search(r"__git_commit__\s*=\s*['\"]([0-9a-f]{7,40})", Path(dist.locate_file(path)).read_text())
            if match:
                return match.group(1)
    return None


def newest_release(name: str, spec: SpecifierSet, allow_prerelease: bool) -> Optional[Version]:
    raw = _open(f"{PYPI}/{name}/json")
    if raw is None:
        return None
    best = None
    for text, files in json.loads(raw)["releases"].items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if version.is_prerelease and not allow_prerelease:
            continue
        if not spec.contains(version, prereleases=True):
            continue
        for f in files:
            requires_python = f.get("requires_python")
            try:
                if requires_python and not SpecifierSet(requires_python).contains("3.12"):
                    continue
            except InvalidSpecifier:
                pass
            if not f.get("yanked") and (f["packagetype"] == "sdist" or _host_wheel(f["filename"])):
                best = version if best is None or version > best else best
                break
    return best


def image_problems(profile: Profile, snapshot: Optional[Path]) -> List[str]:
    dists = installed_dists()
    versions = {name: dist.version for name, dist in dists.items()}
    problems = []
    for name, version in profile.pins().items():
        if versions.get(name) != version:
            problems.append(f"{name}=={version}: the image has {versions.get(name, 'nothing')}")
    commits = {}
    for name, (_, sha) in profile.git_refs().items():
        if name not in dists:
            problems.append(f"{name} @ {sha}: not installed in the image")
            continue
        commits[name] = installed_commit(dists[name])
        if not commit_matches(sha, commits[name]):
            problems.append(f"{name} @ {sha}: the image has commit {commits[name]}")

    # Closure of what the extras install, using the image's metadata.
    specs: Dict[str, List[str]] = {}
    seen: Dict[str, Set[str]] = {}
    queue = list(profile.roots().items())
    while queue:
        name, extras = queue.pop()
        if name in seen and extras <= seen[name]:
            continue
        seen[name] = seen.get(name, set()) | extras
        for raw in (dists[name].requires or []) if name in dists else []:
            req = Requirement(raw)
            if req.marker is not None and not any(req.marker.evaluate({"extra": e}) for e in {""} | seen[name]):
                continue
            dep = canonicalize_name(req.name)
            specs.setdefault(dep, []).append(str(req.specifier))
            queue.append((dep, set(req.extras)))
    overridden = profile.nodeps_names()
    for name in sorted(seen):
        if name not in versions:
            problems.append(f"{name} is needed by the pins but missing from the image")
        elif name not in overridden:
            for spec in specs.get(name, []):
                if spec and not SpecifierSet(spec).contains(versions[name], prereleases=True):
                    problems.append(f"{name}{spec} is required, but the image has {versions[name]}")

    pinned = set(profile.pins()) | set(profile.git_refs())
    unpinned = sorted(name for name in seen if name in versions and name not in pinned and "+" not in versions[name])

    def drift(name: str) -> Optional[str]:
        spec = SpecifierSet(",".join(s for s in specs.get(name, []) if s))
        newest = newest_release(name, spec, Version(versions[name]).is_prerelease)
        if newest is not None and newest != Version(versions[name]):
            return f"pin {name}=={versions[name]}: unpinned, it would resolve to {newest}"
        return None

    with ThreadPoolExecutor(16) as pool:
        problems += [p for p in pool.map(drift, unpinned) if p]

    if snapshot is not None:
        requires = {name: list(dist.requires or []) for name, dist in dists.items()}
        snapshot.write_text(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "versions": versions,
                    "commits": commits,
                    "conflicts": sorted(dependency_conflicts(requires, versions)),
                },
                indent=1,
                sort_keys=True,
            )
        )
    print(f"closure: {len(seen)} packages, {len(unpinned)} unpinned (checked for drift)")
    return problems


def direct_url(dist: metadata.Distribution) -> Dict[str, Any]:
    raw = dist.read_text("direct_url.json")
    return json.loads(raw) if raw else {}


def venv_environment_problems(checkout: Path) -> List[str]:
    import fluxserve

    problems = []
    if sys.prefix == sys.base_prefix:
        problems.append(f"not running in a venv: sys.prefix={sys.prefix}")
    if not Path(fluxserve.__file__).resolve().is_relative_to((checkout / "python").resolve()):
        problems.append(f"fluxserve imports from {fluxserve.__file__}, not {checkout / 'python'}")
    script = shutil.which("fluxserve")
    if script is None or Path(script).parent != Path(sys.prefix) / "bin":
        problems.append(f"the fluxserve command resolves to {script}, not {Path(sys.prefix) / 'bin'}")
    return problems


def torch_problems(profile: Profile, capability: str) -> List[str]:
    import torch

    pin = profile.pins()["torch"]
    problems = []
    if torch.__version__ != pin:
        problems.append(f"torch {torch.__version__} != {pin}")
    if torch.version.cuda != cuda_from_local(pin):
        problems.append(f"torch CUDA {torch.version.cuda} != {cuda_from_local(pin)}")
    expected = tuple(int(part) for part in capability.split("."))
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != expected:
        found = torch.cuda.get_device_capability() if torch.cuda.is_available() else "no GPU"
        problems.append(f"GPU capability {found} != {expected}")
    return problems


def venv_vs_image_problems(profile: Profile, snapshot: Dict[str, Any]) -> List[str]:
    dists = installed_dists()
    refs = profile.git_refs()
    problems = []
    for name, dist in sorted(dists.items()):
        info = direct_url(dist)
        if name == "pip" or info.get("dir_info", {}).get("editable"):
            continue
        image_version = snapshot["versions"].get(name)
        if name in refs:
            commit = info.get("vcs_info", {}).get("commit_id")
            if commit != refs[name][1]:
                problems.append(f"{name}: installed from commit {commit}, pinned {refs[name][1]}")
            if image_version is None or public_version(dist.version) != public_version(image_version):
                problems.append(f"{name}=={dist.version}: the image has {image_version}")
        elif image_version != dist.version:
            problems.append(f"{name}=={dist.version}: the image has {image_version or 'nothing'}")
    for name in sorted(set(refs) - set(dists)):
        problems.append(f"{name}: not installed")
    requires = {name: list(dist.requires or []) for name, dist in dists.items()}
    versions = {name: dist.version for name, dist in dists.items()}
    image_conflicts = {tuple(pair) for pair in snapshot["conflicts"]}
    for dist, dep in sorted(dependency_conflicts(requires, versions) - image_conflicts):
        problems.append(f"dependency conflict the image doesn't have: {dist} -> {dep}")
    return problems


def stamp_problems(checkout: Path, arch: str, torch_version: str) -> List[str]:
    stamps = sorted(checkout.glob("flux-kernel/python/flux_kernel/cuda/*/objs/*.build"))
    problems = [] if len(stamps) >= 4 else [f"expected 4 flux-kernel build stamps, found {len(stamps)}"]
    for stamp in stamps:
        fields = dict(line.split("=", 1) for line in stamp.read_text().splitlines() if "=" in line)
        built = fields.get("arch", fields.get("archs"))
        if built != arch:
            problems.append(f"{stamp.relative_to(checkout)}: built for {built}, expected {arch}")
        if "torch" in fields and fields["torch"] != torch_version:
            problems.append(f"{stamp.relative_to(checkout)}: built with torch {fields['torch']}")
    return problems


def kernel_problems() -> List[str]:
    import torch
    import torch.nn.functional as F
    from flux_kernel import silu_and_mul

    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(8, 256, device="cuda", dtype=torch.bfloat16)
    silu_and_mul(x, out)
    ref = F.silu(x[:, :256]) * x[:, 256:]
    try:
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)  # bf16 rounding
    except AssertionError as exc:
        return [f"silu_and_mul differs from PyTorch: {exc}"]
    return []


def import_problems() -> List[str]:
    import fluxserve

    problems = []
    for info in pkgutil.walk_packages(fluxserve.__path__, "fluxserve.", onerror=lambda name: None):
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] not in OPTIONAL_IMPORTS:
                problems.append(f"{info.name}: No module named {exc.name!r}")
        except Exception as exc:  # reported, not fatal: some modules need a model or a launched server
            print(f"  warning: {info.name}: {type(exc).__name__}: {exc}")
    return problems


def run_checks(checks: List[Tuple[str, Callable[[], List[str]]]]) -> int:
    failed = 0
    for name, check in checks:
        print(f"== {name}", flush=True)
        try:
            problems = check()
        except Exception as exc:
            problems = [f"{type(exc).__name__}: {exc}"]
        for problem in problems:
            print(f"  {problem}")
        print(f"{'FAIL' if problems else 'PASS'}: {name}", flush=True)
        failed += bool(problems)
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    lint = sub.add_parser("lint", help="Static checks")
    lint.add_argument("profiles", nargs="+", type=Path)
    lint.add_argument("--online", action="store_true", help="Also check index availability and git commits")
    image = sub.add_parser("image", help="Compare a profile with the image this python runs in")
    image.add_argument("profile", type=Path)
    image.add_argument("--snapshot", type=Path, help="Write the image snapshot for `venv` here")
    venv = sub.add_parser("venv", help="Compare the venv this python runs in with the profile and image")
    venv.add_argument("profile", type=Path)
    venv.add_argument("--snapshot", type=Path, required=True)
    venv.add_argument("--checkout", type=Path, required=True, help="Repo checkout the venv was installed from")
    venv.add_argument("--stamp-arch", required=True, help="Expected flux-kernel build arch, e.g. 120 or 100a")
    venv.add_argument("--capability", required=True, help="Expected GPU compute capability, e.g. 12.0")
    args = parser.parse_args(argv)

    if args.command == "lint":
        return run_checks(
            [(f"lint {path.name}", lambda path=path: lint_problems(Profile(path), args.online)) for path in args.profiles]
        )
    profile = Profile(args.profile)
    if args.command == "image":
        return run_checks([(f"{profile.path.name} matches the image", lambda: image_problems(profile, args.snapshot))])
    snapshot = json.loads(args.snapshot.read_text())
    return run_checks(
        [
            ("venv and checkout", lambda: venv_environment_problems(args.checkout)),
            ("torch and GPU", lambda: torch_problems(profile, args.capability)),
            ("packages match the profile and the image", lambda: venv_vs_image_problems(profile, snapshot)),
            ("flux-kernel build stamps", lambda: stamp_problems(args.checkout, args.stamp_arch, profile.pins()["torch"])),
            ("flux-kernel silu_and_mul on the GPU", kernel_problems),
            ("every fluxserve module imports", import_problems),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
