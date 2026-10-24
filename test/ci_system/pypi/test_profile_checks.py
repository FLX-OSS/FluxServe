import textwrap
from pathlib import Path

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
    duplicate_names,
    exact_version,
    first_wins,
    lint_problems,
    parse_recipe_args,
    public_version,
    requirement_problem,
)

SHA = "a7d243e0b25697b021cc9366f1326d7b5013287b"
REPO_ROOT = Path(__file__).resolve().parents[3]


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


def write_profile(tmp_path, name, body):
    path = tmp_path / f"pyproject_{name}.toml"
    path.write_text(textwrap.dedent(body))
    return path


def test_profile_collects_pins_git_refs_and_nodeps_groups(tmp_path):
    path = write_profile(
        tmp_path,
        "demo",
        f"""
        [project]
        name = "fluxserve"
        dynamic = ["version", "requires-python"]
        dependencies = []

        [project.optional-dependencies]
        demo = ["torch==2.11.0+cu130", "nvidia-cutlass-dsl[cu13]==4.5.1"]
        dev = ["pytest==9.0.3"]

        [dependency-groups]
        demo-nodeps = [
            "flashinfer-python @ git+https://github.com/FLX-OSS/flashinfer-dllm.git@{SHA}",
            "nvidia-nccl-cu13==2.31.2",
        ]
        """,
    )
    profile = Profile(path)
    assert profile.name == "demo"
    assert profile.pins() == {
        "torch": "2.11.0+cu130",
        "nvidia-cutlass-dsl": "4.5.1",
        "pytest": "9.0.3",
        "nvidia-nccl-cu13": "2.31.2",
    }
    assert profile.git_refs() == {"flashinfer-python": ("https://github.com/FLX-OSS/flashinfer-dllm.git", SHA)}
    assert profile.nodeps_names() == {"flashinfer-python", "nvidia-nccl-cu13"}
    assert profile.roots() == {"torch": set(), "nvidia-cutlass-dsl": {"cu13"}, "pytest": set()}


def test_lint_flags_broken_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_checks, "setuptools_problems", lambda profile: [])
    path = write_profile(
        tmp_path,
        "demo",
        """
        [project]
        name = "fluxserve"
        dynamic = ["version"]
        dependencies = ["torch==2.11.0"]

        [project.optional-dependencies]
        demo = ["torch>=2.11", "numpy==2.3.5", "numpy==2.3.5"]
        """,
    )
    problems = "\n".join(lint_problems(Profile(path), online=False))
    assert "dependencies must stay empty" in problems
    assert "requires-python" in problems
    assert "must be exactly [demo, dev]" in problems
    assert "torch>=2.11: must be an exact == pin" in problems
    assert "numpy is listed more than once" in problems


@pytest.mark.parametrize("name", ["sm120", "sm100"])
def test_repo_profiles_have_no_static_problems(name, monkeypatch):
    # The setuptools config stage and the online checks run in the lint workflow itself.
    monkeypatch.setattr(profile_checks, "setuptools_problems", lambda profile: [])
    assert lint_problems(Profile(REPO_ROOT / f"pyproject_{name}.toml"), online=False) == []
