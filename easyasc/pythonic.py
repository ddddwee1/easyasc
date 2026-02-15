import ast
import inspect
import textwrap
from typing import Any, Optional, cast


class _VarNameAdder(ast.NodeTransformer):
    def __init__(self, target_name: str):
        self._target_name = target_name
        self._in_target = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name != self._target_name or self._in_target:
            return node
        self._in_target = True
        node.decorator_list = [d for d in node.decorator_list if not _is_kernel_decorator(d)]
        new_node = cast(ast.FunctionDef, self.generic_visit(node))
        self._in_target = False
        return new_node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return node

    def visit_Lambda(self, node: ast.Lambda):
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        return node

    def visit_Assign(self, node: ast.Assign):
        if not self._in_target:
            return self.generic_visit(node)
        new_node = cast(ast.Assign, self.generic_visit(node))
        if len(new_node.targets) != 1:
            return new_node
        target = new_node.targets[0]
        value = new_node.value
        self._maybe_add_name_kw(target, value)
        return new_node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if not self._in_target:
            return self.generic_visit(node)
        new_node = cast(ast.AnnAssign, self.generic_visit(node))
        if new_node.value is None:
            return new_node
        self._maybe_add_name_kw(new_node.target, new_node.value)
        return new_node

    def visit_For(self, node: ast.For):
        if not self._in_target:
            return self.generic_visit(node)
        new_node = cast(ast.For, self.generic_visit(node))
        if not isinstance(new_node.iter, ast.Call):
            return new_node
        if not _is_named_call(new_node.iter):
            return new_node
        if _has_name_kw(new_node.iter):
            return new_node
        if isinstance(new_node.target, ast.Name):
            new_node.iter.keywords.append(ast.keyword(arg="name", value=ast.Constant(new_node.target.id)))
        return new_node

    def _maybe_add_name_kw(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) != len(value.elts):
                return
            for t, v in zip(target.elts, value.elts):
                self._maybe_add_name_kw(t, v)
            return
        if not isinstance(target, ast.Name):
            return
        if not isinstance(value, ast.Call):
            return
        if not _is_named_call(value):
            return
        if _has_name_kw(value):
            return
        value.keywords.append(ast.keyword(arg="name", value=ast.Constant(target.id)))


_NAME_CALLS = {"Var", "Tensor", "DBuff", "Reg", "MaskReg", "Min", "CeilDiv", "range", "SEvent", "DEvent", "reinterpret", "split_workspace"}


class _BoolOpRewriter(ast.NodeTransformer):
    def __init__(self, target_name: str):
        self._target_name = target_name
        self._in_target = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name != self._target_name or self._in_target:
            return node
        self._in_target = True
        new_node = cast(ast.FunctionDef, self.generic_visit(node))
        self._in_target = False
        return new_node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return node

    def visit_Lambda(self, node: ast.Lambda):
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        if not self._in_target:
            return self.generic_visit(node)
        new_node = cast(ast.BoolOp, self.generic_visit(node))
        if isinstance(new_node.op, ast.And):
            op: ast.operator = ast.BitAnd()
        elif isinstance(new_node.op, ast.Or):
            op = ast.BitOr()
        else:
            return new_node
        values = new_node.values
        if not values:
            return new_node
        expr = values[0]
        for value in values[1:]:
            expr = ast.BinOp(left=expr, op=op, right=value)
        return ast.copy_location(expr, new_node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not self._in_target:
            return self.generic_visit(node)
        new_node = cast(ast.UnaryOp, self.generic_visit(node))
        if isinstance(new_node.op, ast.Not):
            inverted = ast.UnaryOp(op=ast.Invert(), operand=new_node.operand)
            return ast.copy_location(inverted, new_node)
        return new_node


class _IfRewriter(ast.NodeTransformer):
    def __init__(self, target_name: str):
        self._target_name = target_name
        self._in_target = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name != self._target_name or self._in_target:
            return node
        self._in_target = True
        new_node = cast(ast.FunctionDef, self.generic_visit(node))
        self._in_target = False
        return new_node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return node

    def visit_Lambda(self, node: ast.Lambda):
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        return node

    def visit_If(self, node: ast.If):
        if not self._in_target:
            return self.generic_visit(node)
        def _visit_block(stmts: list[ast.stmt]) -> list[ast.stmt]:
            new_stmts = []
            for stmt in stmts:
                res = self.visit(stmt)
                if isinstance(res, list):
                    new_stmts.extend(res)
                elif res is not None:
                    new_stmts.append(res)
            return new_stmts

        blocks: list[tuple[str, Optional[ast.expr], list[ast.stmt]]] = []
        current = node
        while True:
            blocks.append(("IF", current.test, _visit_block(current.body)))
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            if current.orelse:
                blocks.append(("ELSE", None, _visit_block(current.orelse)))
            break

        stmts = []
        for idx, (kind, test, body) in enumerate(blocks):
            if kind == "ELSE":
                call_name = "Else"
                args: list[ast.expr] = []
            else:
                call_name = "If" if idx == 0 else "Elif"
                args = [cast(ast.expr, test)]
            with_item = ast.withitem(
                context_expr=ast.Call(func=ast.Name(id=call_name, ctx=ast.Load()), args=args, keywords=[]),
                optional_vars=None,
            )
            with_node = ast.With(items=[with_item], body=body, type_comment=None)
            with_node = ast.copy_location(with_node, node)
            stmts.append(with_node)
        return stmts


def _is_named_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _NAME_CALLS
    if isinstance(func, ast.Attribute):
        return func.attr in _NAME_CALLS
    return False


def _has_name_kw(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "name" or kw.arg is None:
            return True
    return False


def _is_kernel_decorator(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id in {"kernel", "func", "auto_sync", "vf"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"kernel", "func", "auto_sync", "vf"}
    return False


def transform_kernel(func):
    source = _get_source(func)
    if source is None:
        return func
    tree = ast.parse(source)
    transformer = _VarNameAdder(func.__name__)
    tree = transformer.visit(tree)
    bool_rewriter = _BoolOpRewriter(func.__name__)
    tree = bool_rewriter.visit(tree)
    if_rewriter = _IfRewriter(func.__name__)
    tree = if_rewriter.visit(tree)
    ast.fix_missing_locations(tree)
    filename = inspect.getsourcefile(func) or "<ast>"
    code = compile(tree, filename=filename, mode="exec")
    temp_ns = {}
    exec(code, func.__globals__, temp_ns)
    new_func = temp_ns.get(func.__name__, func)
    new_func.__defaults__ = func.__defaults__
    new_func.__kwdefaults__ = func.__kwdefaults__
    new_func.__annotations__ = func.__annotations__
    new_func.__doc__ = func.__doc__
    new_func.__module__ = func.__module__
    new_func.__qualname__ = func.__qualname__
    return new_func


def _get_source(func) -> Optional[str]:
    try:
        source = inspect.getsource(func)
    except OSError:
        return None
    return textwrap.dedent(source)
