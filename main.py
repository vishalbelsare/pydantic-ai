from typing import override

import libcst as cst
import libcst.matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor, RemoveImportsVisitor

# 1. Transform `@asynccontextmanager` into `@contextmanager`.
# 2. Transform `async def` into `def`.
# 3. Transform `async with` into `with`.
# 4. Transform `AsyncIterator` into `Iterator`.
# 5. Remove `await`.

# Questions:
# 1. How do we solve MCP Servers?


class AsyncToSyncCommand(VisitorBasedCodemodCommand):
    """A transformer that converts async code to sync."""

    DESCRIPTION = 'Converts async code to sync.'

    def __init__(self, context: CodemodContext) -> None:
        super().__init__(context)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
        if original_node.asynchronous:
            return updated_node.with_changes(asynchronous=None)
        return super().leave_FunctionDef(original_node, updated_node)

    def leave_With(
        self, original_node: cst.With, updated_node: cst.With
    ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
        if original_node.asynchronous:
            return updated_node.with_changes(asynchronous=None)
        return super().leave_With(original_node, updated_node)

    @override
    def leave_For(
        self, original_node: cst.For, updated_node: cst.For
    ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
        if original_node.asynchronous:
            return updated_node.with_changes(asynchronous=None)
        return super().leave_For(original_node, updated_node)

    def leave_Decorator(
        self, original_node: cst.Decorator, updated_node: cst.Decorator
    ) -> cst.Decorator | cst.FlattenSentinel[cst.Decorator] | cst.RemovalSentinel:
        if m.matches(original_node, m.Decorator(decorator=m.Name('asynccontextmanager'))):
            AddImportsVisitor.add_needed_import(self.context, 'contextlib', 'contextmanager')
            RemoveImportsVisitor.remove_unused_import(self.context, 'contextlib', 'asynccontextmanager')
            return updated_node.with_changes(decorator=cst.Name('contextmanager'))
        return super().leave_Decorator(original_node, updated_node)

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
        if original_node.value == 'AsyncIterator':
            AddImportsVisitor.add_needed_import(self.context, 'collections.abc', 'Iterator')
            RemoveImportsVisitor.remove_unused_import(self.context, 'collections.abc', 'AsyncIterator')
            return updated_node.with_changes(value='Iterator')
        return super().leave_Name(original_node, updated_node)

    @m.call_if_inside(m.Await(expression=m.Call()))
    def leave_Await(self, original_node: cst.Await, updated_node: cst.Await) -> cst.BaseExpression:
        return original_node.expression
