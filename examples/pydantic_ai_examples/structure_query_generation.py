from __future__ import annotations

from typing import Any, Literal

from devtools import debug
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ToolOutput

type QueryColumns = Literal[
    'id', 'first_name', 'last_name', 'dob', 'email', 'address', 'phone'
]
table_name = 'users'


class Query(BaseModel, use_attribute_docstrings=True):
    """Structured definition of a database query."""

    select: Literal['*'] | list[SelectColumn] | list[Aggregation] = '*'
    """List of columns to select from the table."""
    filter: FilterClause | FilterAnd | FilterOr | None = None
    """Filter condition for the query."""
    order_by: list[OrderBy] | None = None
    """List of columns to order the results by."""
    limit: int | None = None
    """Maximum number of rows to return."""
    offset: int | None = None
    """Number of rows to skip before returning results."""

    def as_sql(self) -> tuple[str, list[Any]]:
        param_mgr = ParamManager()
        parts: list[str] = ['SELECT']
        if isinstance(self.select, list):
            parts.append('  ' + ', '.join(c.as_sql() for c in self.select))
        else:
            parts.append('  ' + self.select)

        parts.append(f'FROM {table_name}')

        if self.filter:
            parts.append('WHERE ' + self.filter.as_sql(param_mgr))
        if self.order_by:
            parts.append(
                'ORDER BY ' + ', '.join(order_by.as_sql() for order_by in self.order_by)
            )
        if self.limit:
            parts.append(f'LIMIT {self.limit}')
        if self.offset:
            parts.append(f'OFFSET {self.offset}')
        return '\n'.join(parts), param_mgr.params


class SelectColumn(BaseModel):
    name: QueryColumns
    alias: str | None = Field(pattern=r'^[a-zA-Z_][a-zA-Z0-9_ ]*$')

    def as_sql(self) -> str:
        if self.alias:
            return f'{self.name} AS {f'"{self.alias}"' if " " in self.alias else self.alias}'
        else:
            return self.name


class Aggregation(BaseModel):
    column: QueryColumns | Literal['*']
    function: Literal['SUM', 'AVG', 'MAX', 'MIN', 'COUNT']
    alias: str | None = Field(pattern=r'^[a-zA-Z_][a-zA-Z0-9_ ]*$')

    def as_sql(self) -> str:
        func_call = f'{self.function}({self.column})'
        if self.alias:
            return f'{func_call} AS {f'"{self.alias}"' if " " in self.alias else self.alias}'
        else:
            return func_call


class FilterAnd(BaseModel):
    left: FilterClause | FilterAnd | FilterOr
    right: FilterClause | FilterAnd | FilterOr

    def as_sql(self, param_mgr: ParamManager) -> str:
        left_sql = self.left.as_sql(param_mgr)
        right_sql = self.right.as_sql(param_mgr)
        return f'{left_sql} AND {right_sql}'


class FilterOr(BaseModel):
    left: FilterClause | FilterAnd | FilterOr
    right: FilterClause | FilterAnd | FilterOr

    def as_sql(self, param_mgr: ParamManager) -> str:
        left_sql = self.left.as_sql(param_mgr)
        right_sql = self.right.as_sql(param_mgr)
        return f'{left_sql} OR {right_sql}'


class FilterClause(BaseModel):
    column: QueryColumns
    operator: Literal['=', '!=', '<', '>', '<=', '>=']
    value: str | int | float | bool

    def as_sql(self, param_mgr: ParamManager) -> str:
        value_param = param_mgr.add_param(self.value)
        return f'{self.column} {self.operator} {value_param}'


class OrderBy(BaseModel):
    column: QueryColumns
    direction: Literal['ASC', 'DESC']

    def as_sql(self) -> str:
        return f'{self.column} {self.direction}'


class ParamManager:
    def __init__(self) -> None:
        self.params: list[Any] = []

    def add_param(self, param: Any) -> str:
        self.params.append(param)
        return f'${len(self.params)}'


query_agent = Agent(
    'openai:gpt-4o',
    output_type=ToolOutput(type_=Query, name='query'),
    instructions="Generate a query to match the users' preferences.",
)


r = query_agent.run_sync(
    # 'Find user IDs for users who were born before 1990 ordered by how old they are.'
    # 'How many users are there.'
    "what is jane's date of birth."
)
debug(r.output)
query, params = r.output.as_sql()
print(f'SQL:\n-------\n{query}\n-------\n{params=}')
