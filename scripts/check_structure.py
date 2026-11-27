"""Kiểm tra các luật cấu trúc mà ruff và import-linter không bắt được.

Chạy: python scripts/check_structure.py            (lỗi -> exit 1)
      python scripts/check_structure.py --strict   (cảnh báo cũng tính là lỗi)

Luật:
  S1  File .py trong src/ tối đa MAX_LINES dòng (cảnh báo từ WARN_LINES).
  S2  Cấm tên file chung chung / file rác: utils.py, helpers.py, *_old.py, *_v2.py ...
  S3  Mỗi module trong modules/ phải có __init__.py khai báo __all__ (public API).
  S4  Import chéo module chỉ qua public API: `pkg.modules.x`, không `pkg.modules.x.service`.
  S5  Mỗi module nên có thư mục test tương ứng tests/<module>/ (cảnh báo).
  S6  Không để file .py rời ở gốc repo (cảnh báo) -> đưa vào src/ hoặc scripts/.
  S7  Notebook đặt tên dạng NN-ten-ngan.ipynb (cảnh báo).
  S8  Import giữa các module tuân theo thứ tự tầng `layers` trong .devvault.toml.

Cấu hình .devvault.toml (tuỳ chọn, ở gốc repo):
  standard = "1.0"
  profile = "CLI"
  layers = ["reporting", "calls | crm", "ingest"]   # tầng cao trước, "a | b" cùng tầng
  [structure]
  package_dir = "modules"                           # đường dẫn package, tương đối gốc repo
  infra_dirs = ["app", "shared"]                    # bố cục phẳng; mặc định ["app", "shared"]
  [check_structure]
  package = "my_pkg"                                # chọn package khi src/ có nhiều package
  max_lines = 400                                   # S1 (mặc định 400)
  warn_lines = 300                                  # S1 cảnh báo (mặc định 300)
  ignore = ["src/pkg/modules/legacy/big.py:S1"]     # gộp với pyproject.toml

Ngoại lệ (baseline cho code cũ chưa kịp sửa) khai báo trong pyproject.toml:
  [tool.check_structure]
  ignore = ["src/pkg/modules/legacy/big_file.py:S1"]   # "đường-dẫn:LUẬT" hoặc "đường-dẫn"
Mỗi ngoại lệ phải có một dòng Tồn đọng tương ứng trong MOC dự án.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

MAX_LINES = 400
WARN_LINES = 300
BANNED_NAMES = {
    "utils.py",
    "util.py",
    "helpers.py",
    "helper.py",
    "misc.py",
    "common.py",
    "stuff.py",
    "temp.py",
    "tmp.py",
    "new.py",
    "test.py",
    "main2.py",
}
BANNED_PATTERNS = [
    re.compile(p)
    for p in (
        r"_old\.py$",
        r"_v\d+\.py$",
        r"_copy\.py$",
        r"_backup\.py$",
        r"^final.*\.py$",
        r"^untitled",
    )
]
ROOT_PY_ALLOWED = {"setup.py", "conftest.py", "noxfile.py", "manage.py"}
NOTEBOOK_RE = re.compile(r"^\d{2}-[a-z0-9-]+\.ipynb$")
DEFAULT_INFRA_DIRS: tuple[str, ...] = ("app", "shared")


@dataclass
class Finding:
    rule: str
    path: Path
    message: str
    error: bool

    def render(self, root: Path) -> str:
        level = "LỖI " if self.error else "CẢNH BÁO"
        return f"[{level}] {self.rule} {self.path.relative_to(root)}: {self.message}"


@dataclass
class DevVaultConfig:
    profile: str = "-"
    standard: str = "-"
    package: str | None = None
    package_dir: str | None = None
    infra_dirs: tuple[str, ...] = DEFAULT_INFRA_DIRS
    max_lines: int = MAX_LINES
    warn_lines: int = WARN_LINES
    ignore: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Layout:
    pkg: Path
    container: Path
    prefix: str
    infra: frozenset[str] = frozenset()

    @property
    def is_flat(self) -> bool:
        return self.container == self.pkg


def _opt_int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _opt_str(value: object, default: str) -> str:
    return str(value) if value not in (None, "") else default


def _nonempty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _str_tuple(value: object, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not isinstance(value, list):
        return default
    return tuple(item for item in value if isinstance(item, str))


def load_devvault(root: Path) -> DevVaultConfig | None:
    cfg_path = root / ".devvault.toml"
    if not cfg_path.exists():
        return None
    try:
        data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as exc:
        msg = " ".join(str(exc).splitlines())
        sys.stdout.write(f"[CẢNH BÁO] .devvault.toml: {msg}\n")
        return None
    raw_cs = data.get("check_structure")
    cs = raw_cs if isinstance(raw_cs, dict) else {}
    raw_st = data.get("structure")
    st = raw_st if isinstance(raw_st, dict) else {}
    return DevVaultConfig(
        profile=_opt_str(data.get("profile"), "-"),
        standard=_opt_str(data.get("standard"), "-"),
        package=_nonempty_str(cs.get("package")),
        package_dir=_nonempty_str(st.get("package_dir")),
        infra_dirs=_str_tuple(st.get("infra_dirs"), DEFAULT_INFRA_DIRS),
        max_lines=_opt_int(cs.get("max_lines"), MAX_LINES),
        warn_lines=_opt_int(cs.get("warn_lines"), WARN_LINES),
        ignore=_str_tuple(cs.get("ignore")),
        layers=_str_tuple(data.get("layers")),
    )


def find_package(root: Path, name: str | None = None) -> Path | None:
    src = root / "src"
    if not src.is_dir():
        return None
    pkgs = [p for p in src.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
    if name:
        return next((p for p in pkgs if p.name == name), None)
    return pkgs[0] if len(pkgs) == 1 else None


def build_layout(
    pkg: Path,
    *,
    allow_flat: bool = False,
    infra_dirs: tuple[str, ...] = DEFAULT_INFRA_DIRS,
) -> Layout:
    if allow_flat and not (pkg / "modules").is_dir():
        return Layout(pkg=pkg, container=pkg, prefix=f"{pkg.name}.", infra=frozenset(infra_dirs))
    return Layout(pkg=pkg, container=pkg / "modules", prefix=f"{pkg.name}.modules.")


def _resolve_layout(root: Path, cfg: DevVaultConfig) -> tuple[Layout | None, str | None]:
    if cfg.package_dir is not None:
        pkg = root / cfg.package_dir
        if not pkg.is_dir() or not (pkg / "__init__.py").exists():
            msg = (
                f"Không tìm thấy package `{cfg.package_dir}` "
                "(khai ở [structure] package_dir của .devvault.toml).\n"
            )
            return None, msg
        return build_layout(pkg, allow_flat=True, infra_dirs=cfg.infra_dirs), None
    pkg = find_package(root, cfg.package)
    if pkg is None:
        msg = (
            f"Không tìm thấy package `{cfg.package}` trong src/.\n"
            if cfg.package
            else "Không tìm thấy đúng một package trong src/ (src-layout).\n"
        )
        return None, msg
    return build_layout(pkg), None


def check_files(
    pkg: Path,
    max_lines: int = MAX_LINES,
    warn_lines: int = WARN_LINES,
) -> list[Finding]:
    out: list[Finding] = []
    for py in pkg.rglob("*.py"):
        n = len(py.read_text(encoding="utf-8", errors="ignore").splitlines())
        if n > max_lines:
            out.append(Finding("S1", py, f"{n} dòng > {max_lines}, hãy tách file", error=True))
        elif n > warn_lines:
            out.append(Finding("S1", py, f"{n} dòng, sắp chạm giới hạn {max_lines}", error=False))
        name = py.name
        if name in BANNED_NAMES or any(p.search(name) for p in BANNED_PATTERNS):
            out.append(Finding("S2", py, "tên chung chung/file rác, đặt theo mục đích", error=True))
    return out


def declares_all(init: Path) -> bool:
    tree = ast.parse(init.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        )
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            return True
    return False


def module_dirs(layout: Layout) -> list[Path]:
    if not layout.container.is_dir():
        return []
    return sorted(
        d
        for d in layout.container.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and d.name not in layout.infra
    )


def check_public_api(layout: Layout, root: Path) -> list[Finding]:
    out: list[Finding] = []
    for mod in module_dirs(layout):
        init = mod / "__init__.py"
        if not init.exists():
            out.append(Finding("S3", mod, "thiếu __init__.py", error=True))
        elif not declares_all(init):
            out.append(Finding("S3", init, "chưa khai báo __all__ (public API)", error=True))
        if (
            not (root / "tests" / mod.name).is_dir()
            and not (root / "tests" / "unit" / mod.name).is_dir()
        ):
            out.append(Finding("S5", mod, f"chưa có tests/{mod.name}/", error=False))
    return out


def resolve(py: Path, pkg: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    parts = list(py.relative_to(pkg.parent).with_suffix("").parts)
    if node.level > len(parts):
        return None
    base = parts[: len(parts) - node.level]
    if node.module:
        base.append(node.module)
    return ".".join(base)


def owner_module(py: Path, layout: Layout) -> str | None:
    if not py.is_relative_to(layout.container):
        return None
    rel = py.relative_to(layout.container).parts
    if len(rel) <= 1 or rel[0].startswith(("_", ".")) or rel[0] in layout.infra:
        return None
    return rel[0]


def _from_imports(py: Path, layout: Layout, node: ast.ImportFrom) -> list[tuple[int, str, bool]]:
    base = resolve(py, layout.pkg, node)
    if not base:
        return []
    out: list[tuple[int, str, bool]] = [(node.lineno, base, False)]
    if base == layout.prefix[:-1]:
        out += [(node.lineno, f"{base}.{a.name}", False) for a in node.names]
    elif base.startswith(layout.prefix) and base.count(".") == layout.prefix.count("."):
        out += [(node.lineno, f"{base}.{a.name}", True) for a in node.names]
    return out


def imported_names(py: Path, layout: Layout) -> list[tuple[int, str, bool]]:
    """(dòng, tên đầy đủ, là_tên_lấy_từ_module). `from pkg.modules.x import a` -> a là tên."""
    tree = ast.parse(py.read_text(encoding="utf-8"))
    names: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [(node.lineno, a.name, False) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += _from_imports(py, layout, node)
    return names


def public_names(init: Path) -> set[str]:
    """Tên trong __all__ của một __init__.py (rỗng nếu không đọc được)."""
    if not init.exists():
        return set()
    for node in ast.parse(init.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                try:
                    return set(ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    return set()
    return set()


def _check_file_cross_module(
    py: Path,
    layout: Layout,
    known: set[str],
    exported: dict[str, set[str]],
) -> list[Finding]:
    out: list[Finding] = []
    prefix = layout.prefix
    mods = layout.container
    me = owner_module(py, layout)
    for line, name, is_member in imported_names(py, layout):
        if not name.startswith(prefix):
            continue
        parts = name[len(prefix) :].split(".")
        target = parts[0]
        if target == me or target not in known or len(parts) < 2:
            continue
        if is_member and len(parts) == 2 and parts[1] in exported[target]:
            continue  # `from pkg.modules.x import router` với router nằm trong __all__
        if (mods / target / f"{parts[1]}.py").exists() or (mods / target / parts[1]).is_dir():
            msg = f"dòng {line}: import nội bộ `{name}`, hãy import từ `{prefix}{target}`"
            out.append(Finding("S4", py, msg, error=True))
    return out


def check_cross_module(layout: Layout) -> list[Finding]:
    known = {d.name for d in module_dirs(layout)}
    exported = {m: public_names(layout.container / m / "__init__.py") for m in known}
    out: list[Finding] = []
    for py in layout.pkg.rglob("*.py"):
        out.extend(_check_file_cross_module(py, layout, known, exported))
    return out


def check_root(root: Path) -> list[Finding]:
    out = [
        Finding("S6", f, "file .py rời ở gốc repo, chuyển vào src/ hoặc scripts/", False)
        for f in root.glob("*.py")
        if f.name not in ROOT_PY_ALLOWED
    ]
    nb_dir = root / "notebooks"
    if nb_dir.is_dir():
        out += [
            Finding("S7", nb, "đặt tên dạng NN-ten-ngan.ipynb", False)
            for nb in nb_dir.rglob("*.ipynb")
            if ".ipynb_checkpoints" not in nb.parts and not NOTEBOOK_RE.match(nb.name)
        ]
    return out


def load_ignores(root: Path) -> set[str]:
    cfg = root / "pyproject.toml"
    if not cfg.exists():
        return set()
    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    return set(data.get("tool", {}).get("check_structure", {}).get("ignore", []))


def is_ignored(f: Finding, root: Path, ignores: set[str]) -> bool:
    rel = f.path.relative_to(root).as_posix()
    return rel in ignores or f"{rel}:{f.rule}" in ignores


def parse_layers(layers: tuple[str, ...] | list[str]) -> dict[str, int]:
    """Ánh xạ tên module -> số thứ tự tầng (1 = cao nhất)."""
    rank: dict[str, int] = {}
    for idx, entry in enumerate(layers, 1):
        for part in entry.split("|"):
            name = part.strip()
            if name:
                rank.setdefault(name, idx)
    return rank


def _check_file_layers(py: Path, layout: Layout, me: str, rank: dict[str, int]) -> list[Finding]:
    prefix = layout.prefix
    out: list[Finding] = []
    seen: set[tuple[int, str]] = set()
    for line, name, _ in imported_names(py, layout):
        if not name.startswith(prefix):
            continue
        target = name[len(prefix) :].split(".", 1)[0]
        if target == me or target not in rank or (line, target) in seen:
            continue
        seen.add((line, target))
        if rank[target] <= rank[me]:
            msg = (
                f"dòng {line}: module `{me}` (tầng {rank[me]}) "
                f"import `{target}` (tầng {rank[target]}) — trái layers trong .devvault.toml"
            )
            out.append(Finding("S8", py, msg, error=True))
    return out


def check_layers(layout: Layout, root: Path, layers: tuple[str, ...] | list[str]) -> list[Finding]:
    rank = parse_layers(layers)
    if not rank:
        return []
    known = {d.name for d in module_dirs(layout)}
    cfg_path = root / ".devvault.toml"
    out = [
        Finding("S8", cfg_path, f"module `{m}` trong layers không tồn tại", error=False)
        for m in rank
        if m not in known
    ]
    for py in layout.pkg.rglob("*.py"):
        me = owner_module(py, layout)
        if me in rank:
            out.extend(_check_file_layers(py, layout, me, rank))
    return out


def collect_findings(layout: Layout, root: Path, cfg: DevVaultConfig) -> list[Finding]:
    return (
        check_files(layout.pkg, cfg.max_lines, cfg.warn_lines)
        + check_public_api(layout, root)
        + check_cross_module(layout)
        + check_root(root)
        + check_layers(layout, root, cfg.layers)
    )


def _partition_findings(
    findings: list[Finding], root: Path, ignores: set[str]
) -> tuple[list[Finding], int]:
    kept: list[Finding] = []
    skipped = 0
    for f in findings:
        if is_ignored(f, root, ignores):
            skipped += 1
        else:
            kept.append(f)
    return kept, skipped


def _format_header(cfg: DevVaultConfig, layout: Layout) -> str:
    header = f"check_structure · profile: {cfg.profile} · standard: {cfg.standard}"
    if cfg.package_dir:
        suffix = " (phẳng)" if layout.is_flat else ""
        header += f" · package: {cfg.package_dir}{suffix}"
    return header


def main() -> int:
    root = Path.cwd()
    strict = "--strict" in sys.argv
    cfg = load_devvault(root)
    active = cfg or DevVaultConfig()
    layout, err = _resolve_layout(root, active)
    if layout is None:
        sys.stderr.write(err or "")
        return 1
    if cfg is not None:
        sys.stdout.write(_format_header(cfg, layout) + "\n")
    findings, skipped = _partition_findings(
        collect_findings(layout, root, active), root, load_ignores(root) | set(active.ignore)
    )
    for f in sorted(findings, key=lambda item: (not item.error, item.rule, str(item.path))):
        sys.stdout.write(f.render(root) + "\n")
    errors = sum(f.error or strict for f in findings)
    sys.stdout.write(
        f"\ncheck_structure: {errors} lỗi, {len(findings) - errors} cảnh báo, "
        f"{skipped} bỏ qua theo baseline\n"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
