import textwrap

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import profile_checks
from profile_checks import (
    Profile,
    annotation,
    commit_matches,
    cuda_from_local,
    dependency_conflicts,
    dev_closure_problems,
    duplicate_names,
    exact_version,
    first_wins,
    parse_recipe_args,
    profile_problems,
    public_version,
    requirement_problem,
)

SHA = "a7d243e0b25697b021cc9366f1326d7b5013287b"


@pytest.mark.parametrize(
    "raw",
    [
        "torch==2.11.0+cu130",
        "nvidia-cutlass-dsl[cu13]==4.8.0",
        f"flashinfer-python @ git+https://github.com/FLX-OSS/flashinfer-dllm.git@{SHA}",
        f"flash-attn-4[cu13] @ git+https://github.com/Dao-AILab/flash-attention.git@{SHA}#subdirectory=flash_attn/cute",
    ],
)
def test_requirement_problem_accepts_exact_pins_and_sha_refs(raw):
    assert requirement_problem(Requirement(raw)) is None


@pytest.mark.parametrize(
    "raw",
    [
        "torch>=2.11",
        "torch==2.11.*",
        "torch==2.11.0,<3",
        "numpy",
        "flashinfer-python @ git+https://github.com/FLX-OSS/flashinfer-dllm.git@dllm/block-decode",
        f"flashinfer-python @ git+https://github.com/FLX-OSS/flashinfer-dllm.git@{SHA[:12]}",
        f"flashinfer-python @ https://example.com/flashinfer-{SHA}.tar.gz",
        'numpy==2.3.5; python_version >= "3.12"',
    ],
)
def test_requirement_problem_rejects_loose_requirements(raw):
    assert requirement_problem(Requirement(raw)) is not None


def test_exact_version():
    assert exact_version(Requirement("torch==2.11.0+cu130")) == "2.11.0+cu130"
    assert exact_version(Requirement("torch>=2.11")) is None


def test_duplicate_names_use_canonical_names():
    reqs = [Requirement("PyYAML==6.0.1"), Requirement("pyyaml==6.0.3"), Requirement("torch==2.11.0")]
    assert duplicate_names(reqs) == ["pyyaml"]


def test_first_wins_keeps_the_first_copy_on_sys_path():
    kept = first_wins([("setuptools", "/usr/local"), ("Setuptools", "/usr/lib/python3"), ("wheel", "/usr/local")])
    assert kept == {"setuptools": "/usr/local", "wheel": "/usr/local"}


def test_dependency_conflicts_match_pip_check():
    requires = {
        "torch": ['nvidia-nccl-cu13==2.28.9; platform_system == "Linux"', 'optree>=0.13.0; extra == "optree"'],
        "flashinfer-python": ["nccl4py>=0.3.1", "cuda-tile>=1.4.0", "numpy"],
        "flash-attn-4": ["einops"],
    }
    versions = {"nvidia-nccl-cu13": "2.31.2", "cuda-tile": "1.3.0", "numpy": "2.3.5", "einops": "0.8.2"}
    assert dependency_conflicts(requires, versions) == {
        ("torch", "nvidia-nccl-cu13"),
        ("flashinfer-python", "nccl4py"),
        ("flashinfer-python", "cuda-tile"),
    }


def test_dependency_conflicts_allow_prereleases_like_pip_check():
    assert dependency_conflicts({"a": ["b>=4.6.2"]}, {"b": "4.6.3a0"}) == set()


def test_commit_matches_full_and_abbreviated_commits():
    assert commit_matches(SHA, SHA)
    assert commit_matches(SHA, SHA[:7])
    assert not commit_matches(SHA, SHA[:6])
    assert not commit_matches(SHA, "e9515d5dee6ade134a33d6020d38d01ef0596996")
    assert not commit_matches(SHA, None)


def test_public_version_and_cuda_from_local():
    assert public_version("5.17") == public_version("5.17.0")
    assert public_version("2.11.0+cu130") == Version("2.11.0")
    assert public_version("4.0.0b34.dev3+ge9515d5") == public_version("4.0.0b34.dev3+ge9515d5d")
    assert cuda_from_local("2.11.0+cu130") == "13.0"
    assert cuda_from_local("2.11.0") is None


def test_annotation_escapes_workflow_command_characters():
    assert annotation("lint a: b, c", "50% done\npin x==1") == "::error title=lint a%3A b%2C c::50%25 done%0Apin x==1"


def test_parse_recipe_args():
    text = "ARG CUDA_VERSION=13.0.1\nFROM base\nARG TORCH_VERSION=2.11.0\n  ARG INDENTED=1\n"
    assert parse_recipe_args(text) == {"CUDA_VERSION": "13.0.1", "TORCH_VERSION": "2.11.0"}


def write_pyproject(tmp_path, body):
    path = tmp_path / "pyproject.toml"
    path.write_text(textwrap.dedent(body))
    return path


DEMO = f"""
    [project]
    name = "fluxserve"
    dynamic = ["version", "requires-python"]
    dependencies = []

    [project.optional-dependencies]
    sm120 = ["torch==2.11.0+cu130", "nvidia-cutlass-dsl[cu13]==4.8.0", "numpy==2.5.2"]
    sm100 = ["torch==2.11.0+cu130", "nvidia-cutlass-dsl[cu13]==4.5.1", "numpy==2.3.5"]
    sm90 = ["torch==2.11.0+cu130", "numpy==2.5.2"]
    dev = ["scipy==1.18.1", "pytest==9.1.1", "pluggy==1.6.0"]

    [dependency-groups]
    sm100-nodeps = [
        "flashinfer-python @ git+https://github.com/FLX-OSS/flashinfer-dllm.git@{SHA}",
        "nvidia-nccl-cu13==2.31.2",
    ]
    """


def test_profile_collects_its_extra_dev_and_nodeps_group(tmp_path):
    path = write_pyproject(tmp_path, DEMO)
    sm100 = Profile(path, "sm100")
    assert sm100.pins() == {
        "torch": "2.11.0+cu130",
        "nvidia-cutlass-dsl": "4.5.1",
        "numpy": "2.3.5",
        "scipy": "1.18.1",
        "pytest": "9.1.1",
        "pluggy": "1.6.0",
        "nvidia-nccl-cu13": "2.31.2",
    }
    assert sm100.git_refs() == {"flashinfer-python": ("https://github.com/FLX-OSS/flashinfer-dllm.git", SHA)}
    assert sm100.nodeps_names() == {"flashinfer-python", "nvidia-nccl-cu13"}
    assert sm100.roots() == {"torch": set(), "nvidia-cutlass-dsl": {"cu13"}, "numpy": set()}

    sm120 = Profile(path, "sm120")
    assert sm120.pins(dev=False) == {"torch": "2.11.0+cu130", "nvidia-cutlass-dsl": "4.8.0", "numpy": "2.5.2"}
    assert sm120.git_refs() == {} and sm120.nodeps_names() == set()


def test_lint_flags_broken_files_and_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_checks, "setuptools_problems", lambda path: [])
    path = write_pyproject(
        tmp_path,
        """
        [project]
        name = "fluxserve"
        dynamic = ["version"]
        dependencies = ["torch==2.11.0"]

        [project.optional-dependencies]
        sm120 = ["torch>=2.11", "numpy==2.3.5", "numpy==2.3.5"]
        sm100 = ["pyyaml==6.0.1"]
        dev = ["pyyaml==6.0.3"]

        [dependency-groups]
        sm80-nodeps = ["numpy==2.3.5"]
        """,
    )
    problems = "\n".join(profile_checks.file_problems(path))
    assert "dependencies must stay empty" in problems
    assert "requires-python" in problems
    assert "must be exactly ['dev', 'sm100', 'sm120', 'sm90']" in problems
    assert "['sm80-nodeps'] must be named <platform>-nodeps" in problems
    sm120 = "\n".join(profile_problems(Profile(path, "sm120"), online=False))
    assert "torch>=2.11: must be an exact == pin" in sm120
    assert "numpy is listed more than once" in sm120
    assert "pyyaml is listed more than once" in "\n".join(profile_problems(Profile(path, "sm100"), online=False))


def test_dev_closure_must_be_pinned_compatibly(tmp_path, monkeypatch):
    requires = {
        ("scipy", "1.18.1"): ["numpy<2.8,>=2.4"],
        ("pytest", "9.1.1"): ["pluggy<2,>=1.5", "iniconfig>=1.0.1", 'colorama>=0.4; sys_platform == "win32"'],
        ("pluggy", "1.6.0"): ['pytest; extra == "testing"'],
    }
    monkeypatch.setattr(profile_checks, "pypi_requires", lambda name, version: tuple(requires[name, version]))
    path = write_pyproject(tmp_path, DEMO)
    assert dev_closure_problems(Profile(path, "sm120")) == ["iniconfig>=1.0.1 (needed by pytest==9.1.1) is not pinned"]
    assert dev_closure_problems(Profile(path, "sm100")) == [
        "iniconfig>=1.0.1 (needed by pytest==9.1.1) is not pinned",
        "scipy==1.18.1 requires numpy<2.8,>=2.4, but numpy is pinned to 2.3.5",
    ]


@pytest.mark.parametrize("name", profile_checks.PLATFORMS)
def test_repo_profiles_have_no_static_problems(name):
    # The setuptools config stage and the online checks run in the lint workflow itself.
    assert profile_problems(Profile(profile_checks.PYPROJECT, name), online=False) == []


def test_repo_pyproject_has_no_static_problems(monkeypatch):
    monkeypatch.setattr(profile_checks, "setuptools_problems", lambda path: [])
    assert profile_checks.file_problems(profile_checks.PYPROJECT) == []
