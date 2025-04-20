import ast
import inspect
from collections.abc import Awaitable, Sequence
from typing import Any, Callable, Literal

from .mcp import MCPServerStdio

__all__ = ('mcp_run_python_stdio',)

MCP_RUN_PYTHON_VERSION = '0.0.13'
Callback = Callable[..., Awaitable[Any]]


def mcp_run_python_stdio(callbacks: Sequence[Callback] = (), *, local_code: bool = False) -> MCPServerStdio:
    """Prepare a server server connection using `'stdio'` transport.

    Args:
        callbacks: A sequence of callback functions to be register on the server.
        local_code: Whether to run local `mcp-run-python` code.

    Returns:
        A server connection definition.
    """
    return MCPServerStdio(
        'deno',
        args=_deno_args('stdio', callbacks, local_code),
        cwd='mcp-run-python' if local_code else None,
    )


def _deno_args(mode: Literal['stdio', 'sse'], callbacks: Sequence[Callback], local_code: bool) -> list[str]:
    args = [
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'src/main.ts' if local_code else f'jsr:@pydantic/mcp-run-python@{MCP_RUN_PYTHON_VERSION}',
        mode,
    ]

    if callbacks:
        sigs = '\n\n'.join(_callback_signature(cb) for cb in callbacks)
        args += ['--callbacks', sigs]
    return args


def _callback_signature(func: Callback) -> str:
    """Extract the signature of a function.

    This simply means getting the source code of the function, and removing the body of the function while keeping the docstring.
    """
    source = inspect.getsource(func)
    ast_mod = ast.parse(source)
    assert isinstance(ast_mod, ast.Module), f'Expected Module, got {type(ast_mod)}'
    assert len(ast_mod.body) == 1, f'Expected single function definition, got {len(ast_mod.body)}'
    f = ast_mod.body[0]
    assert isinstance(f, ast.AsyncFunctionDef), f'Expected an async function, got {type(func)}'
    lines = lines = source.splitlines()
    e = f.body[0]
    # if the first expression is a docstring, keep it and no need for an ellipsis as the body
    if isinstance(e, ast.Expr) and isinstance(e.value, ast.Constant) and isinstance(e.value.value, str):
        e = f.body[1]
        lines = lines[: e.lineno - 1]
    else:
        lines = lines[: e.lineno - 1]
        lines.append(e.col_offset * ' ' + '...')
    return '\n'.join(lines)
