"""Generate .pyi stubs for pybind11 extensions via pybind11-stubgen.

Types, classes, inheritance, properties, submodules, aliases, and constants come
solely from pybind11-stubgen reflecting the compiled extension binary. When
tree-sitter-cpp is available and csrc_root is provided, an optional AST pass fills
missing docstrings from leading C++ comments and replaces unnamed positional
parameters (`arg0`, `arg1`, ...) with C++ parameter names marked `/`.
"""

from __future__ import annotations

import ast
import importlib.util
import keyword
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

CPP_EXTS = {'.hpp', '.h', '.cpp', '.cc', '.cxx'}
TYPE_SUBS = (
    (re.compile(r'\btyping\.SupportsInt\s*\|\s*typing\.SupportsIndex\b'), 'int'),
    (re.compile(r'\btyping\.SupportsFloat\s*\|\s*typing\.SupportsIndex\b'), 'float'),
)
NON_DOC_PREFIXES = ('TODO', 'FIXME', 'NOLINT', 'clang-format', 'oss-ignore', 'ReSharper', 'noinspection')
DOC_PARENTS = (
    'translation_unit', 'declaration_list', 'field_declaration_list', 'compound_statement',
    'namespace_definition', 'preproc_if', 'preproc_ifdef', 'preproc_else', 'preproc_elif',
)
_SUBPROCESS_CODE = (
    'import importlib, sys; '
    'so_dir, mod, out = sys.argv[1:4]; '
    '[importlib.import_module(m) for m in sys.argv[4:]]; '
    'from pybind11_stubgen import main; '
    'sys.path.insert(0, so_dir); '
    'sys.argv = ["pybind11-stubgen", mod, "-o", out]; main()'
)

CppKey = Tuple[Optional[str], str, int]  # (cpp_class_or_None, func_name, num_params)
CppMeta = Tuple[Optional[List[str]], Optional[str]]  # (param_names_or_None_if_ambiguous, doc)


def _split_top_level(s: str) -> list[str]:
    parts, cur, depth, quote, escape = [], [], 0, None, False
    for ch in s:
        if quote is not None:
            cur.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            cur.append(ch)
            continue
        depth += (ch in '([{') - (ch in ')]}')
        if ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return [p for p in parts if p]


def _strip_template_args(s: str) -> str:
    out, depth = [], 0
    for ch in s:
        if ch == '<':
            depth += 1
        elif ch == '>' and depth > 0:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return ''.join(out)


def _extract_doc(node, *, climb: bool = True) -> str | None:
    if climb:
        while node.parent and node.parent.type not in DOC_PARENTS:
            node = node.parent
    blocks, expected_row, sib = [], node.start_point[0], node.prev_sibling
    while sib and sib.type == 'comment':
        if sib.end_point[0] + 1 < expected_row:
            break
        prev = sib.prev_sibling
        if prev is not None and prev.type != 'comment' and prev.end_point[0] == sib.start_point[0]:
            break
        raw = sib.text.decode('utf-8', errors='ignore')
        if raw.startswith('//'):
            blocks.append([re.sub(r'^//+\s?', '', raw).rstrip()])
        else:
            inner = re.sub(r'^/\*+\s?|\s*\*+/$', '', raw, flags=re.S)
            blocks.append([re.sub(r'^\s*\*\s?', '', ln).rstrip() for ln in inner.splitlines()])
        expected_row = sib.start_point[0]
        sib = prev
    lines = [ln for blk in reversed(blocks) for ln in blk]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty or all(ln.lstrip().startswith(NON_DOC_PREFIXES) for ln in non_empty):
        return None
    return '\n'.join(lines)


def _extract_cpp_ast_metadata(
    csrc_root: Path,
) -> tuple[dict[CppKey, CppMeta], dict[tuple[str | None, str], tuple[str | None, str]], dict[str, str]]:
    """Extract scoped (param_names, doc) metadata via tree-sitter-cpp AST."""
    if not csrc_root.is_dir():
        return {}, {}, {}
    try:
        import tree_sitter_cpp
        from tree_sitter import Language, Parser, Query, QueryCursor
    except ImportError:
        return {}, {}, {}

    lang = Language(tree_sitter_cpp.language())
    parser = Parser(lang)
    q_class = Query(lang, '[(class_specifier name: (type_identifier) @name) (struct_specifier name: (type_identifier) @name)]')
    q_func = Query(lang, '(function_declarator declarator: (_) @decl parameters: (parameter_list) @params)')
    q_alias = Query(lang, '(init_declarator declarator: (identifier) @name value: [(identifier) (template_function)] @val)')
    q_call = Query(lang, '(call_expression function: (field_expression field: (field_identifier) @method) arguments: (argument_list) @args)')

    def run_query(q, root):
        return [m for _, m in QueryCursor(q).matches(root)]

    def text(n) -> str:
        return n.text.decode('utf-8', errors='ignore')

    def has_ancestor(node, types: tuple[str, ...]):
        while (node := node.parent) is not None:
            if node.type in types:
                return node
        return None

    def leaf_ident(node) -> str | None:
        if node is None:
            return None
        if node.type in ('identifier', 'field_identifier'):
            return text(node)
        return next((r for c in node.children if (r := leaf_ident(c))), None)

    def param_names_from(plist) -> list[str]:
        if plist is None:
            return []
        decls = [p for p in plist.named_children if p.type in ('parameter_declaration', 'optional_parameter_declaration')]
        if len(decls) == 1 and text(decls[0]).strip() == 'void':
            return []
        return [n if (n := leaf_ident(p.child_by_field_name('declarator')) or '').isidentifier() and not keyword.iskeyword(n) else ''
                for p in decls]

    def enclosing_class(node) -> str | None:
        cls = has_ancestor(node, ('class_specifier', 'struct_specifier'))
        name = cls.child_by_field_name('name') if cls is not None else None
        return text(name) if name is not None else None

    def find_template_fn(node):
        if node is None or node.type == 'template_function':
            return node
        return next((r for c in node.named_children if (r := find_template_fn(c)) is not None), None)

    def class_chain_info(call_node) -> tuple[str, str] | None:
        cur = call_node
        while True:
            fn = cur.child_by_field_name('function')
            if fn is None or fn.type != 'field_expression':
                return None
            obj = fn.child_by_field_name('argument')
            inner = obj.child_by_field_name('function') if obj.type == 'call_expression' else None
            if inner is not None and inner.type == 'field_expression':
                cur = obj
                continue
            if obj.type != 'call_expression':
                return None
            tf = find_template_fn(obj.child_by_field_name('function'))
            if tf is None or text(tf.child_by_field_name('name')).split('::')[-1] != 'class_':
                return None
            targs, cargs = tf.child_by_field_name('arguments'), obj.child_by_field_name('arguments')
            if not targs or not targs.named_children or not cargs:
                return None
            cpp_cls = _strip_template_args(text(targs.named_children[0])).split('::')[-1].strip()
            py_cls = next((text(a).strip('"') for a in cargs.named_children if a.type == 'string_literal'), cpp_cls)
            return py_cls, cpp_cls

    roots = [parser.parse(p.read_bytes()).root_node
             for p in sorted(csrc_root.rglob('*')) if p.is_file() and p.suffix.lower() in CPP_EXTS]
    known_classes = {text(m['name'][0]) for root in roots for m in run_query(q_class, root)}

    cpp_funcs: dict[CppKey, CppMeta] = {}
    ambiguous: set[CppKey] = set()
    aliases: dict[tuple[str | None, str], str] = {}
    py_to_cpp: dict[tuple[str | None, str], tuple[str | None, str]] = {}
    py_cls_to_cpp: dict[str, str] = {}

    def record(key: CppKey, pnames: list[str], doc: str | None) -> None:
        if key not in cpp_funcs:
            cpp_funcs[key] = (pnames, doc)
            return
        prev_params, prev_doc = cpp_funcs[key]
        if key in ambiguous or (prev_params is not None and prev_params != pnames):
            ambiguous.add(key)
            prev_params = None
        cpp_funcs[key] = (prev_params, prev_doc or doc)

    for root in roots:
        for m in run_query(q_func, root):
            decl = m['decl'][0]
            stmt = has_ancestor(decl, ('function_definition', 'declaration', 'field_declaration'))
            if stmt is None or has_ancestor(stmt, ('compound_statement',)) is not None:
                continue

            if decl.type == 'qualified_identifier':
                parts = [p.strip() for p in _strip_template_args(text(decl)).split('::') if p.strip()]
                fn_name = parts[-1] if parts else ''
                raw_scope = parts[-2] if len(parts) >= 2 else None
                cls_name = raw_scope if (raw_scope in known_classes or raw_scope == fn_name) else enclosing_class(stmt)
            else:
                cls_name, fn_name = enclosing_class(stmt), _strip_template_args(text(decl)).strip()
            if not fn_name or fn_name.startswith('~'):
                continue
            if cls_name and fn_name == cls_name:
                fn_name = '__init__'
            pnames = param_names_from(m['params'][0])
            record((cls_name, fn_name, len(pnames)), pnames, _extract_doc(stmt))

        for m in run_query(q_alias, root):
            name_node = m['name'][0]
            if has_ancestor(name_node, ('compound_statement',)) is not None:
                continue
            target = _strip_template_args(text(m['val'][0])).split('::')[-1].strip()
            aliases[(enclosing_class(name_node), text(name_node))] = target

        for m in run_query(q_call, root):
            method = text(m['method'][0])
            if method not in ('def', 'def_static'):
                continue
            args_node = m['args'][0]
            args = [c for c in args_node.named_children if c.type != 'comment']
            if not args:
                continue
            cls_info = class_chain_info(args_node.parent)
            py_cls = cls_info[0] if cls_info else None
            if cls_info:
                py_cls_to_cpp[cls_info[0]] = cls_info[1]
            if args[0].type != 'string_literal' or len(args) < 2:
                continue
            py_name, target = text(args[0]).strip('"'), args[1]
            if target.type == 'lambda_expression':
                ldecl = target.child_by_field_name('declarator')
                pnames = param_names_from(ldecl.child_by_field_name('parameters') if ldecl else None)
                if method == 'def' and py_cls is not None and pnames:
                    pnames = pnames[1:]
                record((py_cls, py_name, len(pnames)), pnames, _extract_doc(target, climb=(py_cls is None)))
            else:
                if target.type == 'call_expression':
                    fn_node = target.child_by_field_name('function')
                    call_args_node = target.child_by_field_name('arguments')
                    if fn_node is not None and 'overload_cast' in text(fn_node) and call_args_node is not None:
                        inner_args = [c for c in call_args_node.named_children if c.type != 'comment']
                        if inner_args:
                            target = inner_args[0]
                parts = [p.strip() for p in _strip_template_args(text(target).lstrip('&')).split('::') if p.strip()]
                if not parts:
                    continue
                cpp_name = parts[-1]
                raw_scope = parts[-2] if len(parts) >= 2 else None
                cpp_scope = raw_scope if raw_scope in known_classes else (cls_info[1] if cls_info else None)
                if cpp_name.isidentifier():
                    py_to_cpp[(py_cls, py_name)] = (cpp_scope, cpp_name)

    for (scope, alias), target in aliases.items():
        seen = {alias}
        while (scope, target) in aliases and target not in seen:
            seen.add(target)
            target = aliases[(scope, target)]
        for (k_scope, k_name, argc), meta in list(cpp_funcs.items()):
            if k_scope == scope and k_name == target:
                cpp_funcs.setdefault((scope, alias, argc), meta)

    return cpp_funcs, py_to_cpp, py_cls_to_cpp


def _has_unresolved_cpp_annotation(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        anns = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                anns.append(node.returns)
            anns.extend(a.annotation for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs if a.annotation is not None)
        elif isinstance(node, ast.AnnAssign):
            anns.append(node.annotation)
        elif isinstance(node, ast.ClassDef):
            anns.extend(node.bases)
        if any(
            isinstance(sub, ast.Constant) and isinstance(sub.value, str) and '::' in sub.value
            for a in anns for sub in ast.walk(a)
        ):
            return True
    return False


def _is_nullable_type(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name) and node.id in ('Any', 'object'):
        return True
    if isinstance(node, ast.Attribute) and (
        node.attr == 'Any' or (isinstance(node.value, ast.Name) and node.value.id == 'builtins' and node.attr == 'object')
    ):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_nullable_type(node.left) or _is_nullable_type(node.right)
    if isinstance(node, ast.Subscript):
        base = node.value.attr if isinstance(node.value, ast.Attribute) else (
            node.value.id if isinstance(node.value, ast.Name) else ''
        )
        if base == 'Optional':
            return True
        if base == 'Union':
            sl = node.slice.value if hasattr(ast, 'Index') and isinstance(node.slice, ast.Index) else node.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            return any(_is_nullable_type(e) for e in elts)
    return False


def _normalize_nullable_default(part: str) -> str:
    m = re.match(r'^([a-zA-Z_]\w*)\s*:\s*(.+?)\s*=\s*None\s*$', part)
    if not m:
        return part
    name, ann = m.group(1), m.group(2).strip()
    try:
        node = ast.parse(ann, mode='eval').body
    except SyntaxError:
        return part
    return part if _is_nullable_type(node) else f'{name}: {ann} | None = None'


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _postprocess_pyi(
    pyi_path: Path,
    cpp_funcs: dict[CppKey, CppMeta],
    py_to_cpp: dict[tuple[str | None, str], tuple[str | None, str]],
    py_cls_to_cpp: dict[str, str],
) -> None:
    raw_text = pyi_path.read_text(encoding='utf-8')
    text = raw_text
    for pat, repl in TYPE_SUBS:
        text = pat.sub(repl, text)
    stripped_typing = re.sub(r'^import typing\n', '', text, flags=re.M)
    if 'typing.' not in stripped_typing:
        text = stripped_typing

    lines, out = text.splitlines(), []
    class_re = re.compile(r'^class ([a-zA-Z_]\w*)\b')
    def_re = re.compile(r'^(\s*)def ([a-zA-Z_]\w*)\((.*)\) -> (.+):$')
    current_class: str | None = None
    i = 0

    while i < len(lines):
        if cm := class_re.match(lines[i]):
            current_class = cm.group(1)
        elif lines[i] and not lines[i][0].isspace():
            current_class = None

        m = def_re.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        indent, fn_name, params_str, ret_str = m.groups()
        parts = _split_top_level(params_str) if params_str.strip() else []
        offset = 1 if (parts and parts[0] in ('self', 'cls')) else 0
        arg_parts = parts[offset:]
        argc = sum(1 for p in arg_parts if p not in ('/', '*'))

        cpp_cls = py_cls_to_cpp.get(current_class, current_class) if current_class else None
        meta = None
        if (current_class, fn_name) in py_to_cpp:
            t_scope, t_name = py_to_cpp[(current_class, fn_name)]
            meta = cpp_funcs.get((t_scope, t_name, argc)) or (
                cpp_funcs.get((None, t_name, argc)) if current_class is None else None
            )
        if meta is None:
            meta = cpp_funcs.get((cpp_cls, fn_name, argc)) or cpp_funcs.get((current_class, fn_name, argc))

        changed = False
        if meta and meta[0] is not None and re.search(r'\barg0\s*:', params_str):
            c_names = meta[0]
            if len(c_names) == argc and all(c_names):
                used_names = {
                    p.split(':', 1)[0].split('=', 1)[0].strip()
                    for p in parts if p not in ('/', '*')
                }
                renamed, last_pos_only, c_iter = list(parts[:offset]), -1, iter(c_names)
                for part in arg_parts:
                    if part in ('/', '*'):
                        renamed.append(part)
                        continue
                    cname = next(c_iter)
                    if (am := re.match(r'^arg\d+(\s*:.*)$', part)) and cname not in used_names:
                        renamed.append(f'{cname}{am.group(1)}')
                        used_names.add(cname)
                        last_pos_only = len(renamed) - 1
                    else:
                        renamed.append(part)
                if last_pos_only != -1 and '/' not in renamed:
                    renamed.insert(last_pos_only + 1, '/')
                parts = renamed
                changed = True

        normalized = [_normalize_nullable_default(p) for p in parts]
        if normalized != parts:
            parts = normalized
            changed = True
        if changed:
            params_str = ', '.join(parts)

        out.append(f'{indent}def {fn_name}({params_str}) -> {ret_str}:')
        i += 1

        if meta and meta[1] and i < len(lines) and lines[i].strip() == '...':
            doc = meta[1].replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
            body_indent = indent + '    '
            if '\n' in doc or doc.endswith('"'):
                doc_lines = '\n'.join(f'{body_indent}{ln}'.rstrip() for ln in doc.splitlines())
                out.append(f'{body_indent}"""\n{doc_lines}\n{body_indent}"""')
            else:
                out.append(f'{body_indent}"""{doc}"""')
            out.append(lines[i])
            i += 1

    final = '\n'.join(out) + '\n'
    try:
        tree = ast.parse(final)
        compile(tree, pyi_path.name, 'exec')
    except SyntaxError as exc:
        print(f'[stubgen] warning: post-processing produced invalid syntax in {pyi_path.name} ({exc}); keeping raw stub')
        final = raw_text
        tree = ast.parse(final)
    if _has_unresolved_cpp_annotation(tree):
        print(f'[stubgen] warning: unresolved C++ symbol ("::") detected in {pyi_path.name}; check pybind11 registration order')
    pyi_path.write_text(final, encoding='utf-8')


def generate_stubs(
    so_dir: str | Path,
    module_name: str = '_C',
    output_dir: str | Path | None = None,
    csrc_root: str | Path | None = None,
    preload: Iterable[str] = ('torch',),
) -> bool:
    """Generate .pyi stubs for a compiled pybind11 extension. Returns True if generated."""
    if importlib.util.find_spec('pybind11_stubgen') is None:
        print('[stubgen] pybind11-stubgen not installed; skipping .pyi generation')
        return False

    so_dir = Path(so_dir).resolve()
    out_root = Path(output_dir).resolve() if output_dir else so_dir
    out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=out_root, prefix=f'.{module_name}.stubgen.') as tmp:
        cmd = [sys.executable, '-c', _SUBPROCESS_CODE, str(so_dir), module_name, tmp, *tuple(preload)]
        subprocess.run(cmd, check=True, timeout=60)

        gen_file, gen_dir = Path(tmp) / f'{module_name}.pyi', Path(tmp) / module_name
        if gen_dir.is_dir():
            pyi_files, src_artifact, dst_artifact = list(gen_dir.rglob('*.pyi')), gen_dir, out_root / module_name
            stale_sibling = out_root / f'{module_name}.pyi'
        elif gen_file.is_file():
            pyi_files, src_artifact, dst_artifact = [gen_file], gen_file, out_root / f'{module_name}.pyi'
            stale_sibling = out_root / module_name
        else:
            raise RuntimeError(f'pybind11-stubgen produced neither {gen_file} nor {gen_dir}')

        # Post-process and validate inside same-filesystem staging dir before publishing.
        cpp_funcs, py_to_cpp, py_cls_to_cpp = _extract_cpp_ast_metadata(Path(csrc_root)) if csrc_root else ({}, {}, {})
        for p in pyi_files:
            _postprocess_pyi(p, cpp_funcs, py_to_cpp, py_cls_to_cpp)

        if src_artifact.is_file():
            os.replace(src_artifact, dst_artifact)
        else:
            backup = Path(tmp) / '_prev'
            had_prev = dst_artifact.is_symlink() or dst_artifact.exists()
            if had_prev:
                os.rename(dst_artifact, backup)
            try:
                os.rename(src_artifact, dst_artifact)
            except Exception:
                if had_prev and (backup.is_symlink() or backup.exists()):
                    os.rename(backup, dst_artifact)
                raise
        _remove_path(stale_sibling)
        print(f'[stubgen] generated {dst_artifact}')
        return True


def make_build_ext(base_cls, *, csrc_root: str | Path | None = 'csrc', preload: Iterable[str] = ('torch',)):
    """Wrap a setuptools/torch build_ext command class to emit .pyi stubs after compiling extensions."""
    class StubBuildExt(base_cls):
        def run(self):
            super().run()
            require_stubs = os.environ.get('DEEP_JIT_REQUIRE_STUBS', '0') == '1'
            for ext in (self.extensions or ()):
                ext_path = Path(self.get_ext_fullpath(ext.name))
                module_name = ext.name.rsplit('.', 1)[-1]
                try:
                    ok = generate_stubs(ext_path.parent, module_name, ext_path.parent, csrc_root, preload)
                    if require_stubs and not ok:
                        raise RuntimeError(f'DEEP_JIT_REQUIRE_STUBS=1 but pybind11-stubgen is unavailable for {ext.name}')
                except Exception as exc:
                    if require_stubs:
                        raise
                    print(f'[stubgen] warning: failed to generate stubs for {ext.name}: {exc}')

    StubBuildExt.__name__ = f'Stub{base_cls.__name__}'
    return StubBuildExt
