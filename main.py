import ast
from pathlib import Path
from typing import Any


class AsyncToSyncTransformer(ast.NodeTransformer):
    """A transformer that converts async code to sync."""

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # Create a new FunctionDef node with the same attributes as the AsyncFunctionDef node
        new_decorator_list: list[ast.expr] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'asynccontextmanager':
                decorator.id = 'contextmanager'
            new_decorator_list.append(decorator)
        new_node = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=new_decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
            type_params=node.type_params,
            **_extract_attributes(node),
        )
        self.generic_visit(new_node)
        return new_node

    def visit_AsyncWith(self, node: ast.AsyncWith):
        new_node = ast.With(
            items=node.items,
            body=node.body,
            type_comment=node.type_comment,
            **_extract_attributes(node),
        )
        self.generic_visit(new_node)
        return new_node

    def visit_AsyncFor(self, node: ast.AsyncFor):
        return self.generic_visit(node)

    def visit_Await(self, node: ast.Await):
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == 'contextlib':
            for idx, alias in enumerate(node.names.copy()):
                # Repeated imports are removed when ruff runs.
                if alias.name in 'asynccontextmanager':
                    node.names[idx].name = 'contextmanager'
        return node


def _extract_attributes(node: ast.stmt) -> dict[str, Any]:
    return dict(
        lineno=node.lineno,
        col_offset=node.col_offset,
        end_lineno=node.end_lineno,
        end_col_offset=node.end_col_offset,
    )


code = Path('pydantic_ai_slim/pydantic_ai/agent.py').read_text()
tree = ast.parse(code)

visitor = AsyncToSyncTransformer()
transformed_tree = visitor.visit(tree)
Path('agent.py').write_text(ast.unparse(transformed_tree))
