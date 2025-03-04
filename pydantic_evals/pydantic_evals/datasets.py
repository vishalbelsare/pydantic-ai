from __future__ import annotations as _annotations

import asyncio
import sys
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Generic, Self

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_core import to_json, to_jsonable_python
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from pydantic_ai import models

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])

DEFAULT_DATASET_PATH = './test_cases.yaml'


class DatasetRow(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    """A single row of a "dataset", consisting of input, expected output, and metadata."""

    name: str
    inputs: InputsT
    metadata: MetadataT
    expected_output: OutputT | None = None


class Dataset(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    """A dataset of test cases, each consisting of input, expected output, and metadata."""

    rows: list[DatasetRow[InputsT, OutputT, MetadataT]]

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_DATASET_PATH) -> Self:
        path = cls._get_relative_path(path)
        if not path.exists():
            raise FileNotFoundError(f'{cls.__name__} dataset file {path} does not exist')

        raw = path.read_text()
        loaded = yaml.safe_load(raw)
        try:
            return cls.model_validate(loaded)
        except ValidationError as e:
            raise ValueError(
                f'{cls.__name__} dataset file {path} contains data that does not match the schema:\n{e}.'
            ) from e

    @classmethod
    def generate_dataset_files(
        cls,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        schema_path: Path | str | None = None,
    ) -> str:
        dataset_path = cls._get_relative_path(dataset_path)

        if schema_path is None:
            if dataset_path.exists():
                # Try to infer the schema path from the first line of the existing dataset file
                first_line = dataset_path.read_text().split('\n', 1)[0]
                if first_line.startswith('# yaml-language-server: $schema='):
                    schema_path = (dataset_path.parent / first_line.split('=', 1)[1]).resolve()
            if schema_path is None:
                schema_path = dataset_path.parent / f'./{dataset_path.stem}_schema.json'
        else:
            schema_path = cls._get_relative_path(schema_path)

        schema_content = to_json(cls.model_json_schema(), indent=2).decode() + '\n'
        if not schema_path.exists() or schema_path.read_text() != schema_content:
            schema_path.write_text(schema_content)

        schema_ref = str(_get_relative_path_reference(schema_path, dataset_path.parent))
        yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
        if dataset_path.exists():
            try:
                cls.from_yaml(dataset_path)
            except ValueError as e:
                if isinstance(e.__cause__, ValidationError):
                    raise ValueError(
                        f'{cls.__name__} dataset file {dataset_path} already exists, but does not contain compatible data.'
                        f' Fix or delete the file before calling this function:\n{e.__cause__}'
                    ) from e.__cause__
                else:
                    raise
            dataset_text = dataset_path.read_text()
            cases_text_with_schema = _ensure_yaml_language_server_line(dataset_text, schema_ref)
            if cases_text_with_schema != dataset_text:
                dataset_path.write_text(cases_text_with_schema)
        else:
            content = yaml.dump(to_jsonable_python(cls(rows=[])), sort_keys=False)
            dataset_path.write_text(f'{yaml_language_server_line}\n{content}')
        return schema_ref

    @classmethod
    def generate_dataset_examples(
        cls,
        model: models.Model | models.KnownModelName = 'gpt-4o',
        min_count: int = 3,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
    ):
        dataset_path = cls._get_relative_path(dataset_path)
        schema_ref = cls.generate_dataset_files(dataset_path=dataset_path, schema_path=None)

        existing_content: str | None = None

        try:
            existing_rows = cls.from_yaml(dataset_path).rows
            min_count = max(0, min_count - len(existing_rows))
            if min_count == 0:
                return  # nothing to do, already have enough examples
            if existing_rows:
                existing_content = dataset_path.read_text()
        except FileNotFoundError:
            pass  # in this case, we'll generate a new file, so we ignore the error

        examples = asyncio.run(_generate_examples(cls, dataset_path, model, min_count))

        if existing_content is None:
            content = yaml.dump(to_jsonable_python(cls(rows=examples)), sort_keys=False)
            content = _ensure_yaml_language_server_line(content, schema_ref)
            dataset_path.write_text(content)
        else:
            new_lines = yaml.dump(to_jsonable_python(cls(rows=examples)), sort_keys=False).splitlines()
            new_lines = new_lines[1:]  # drop the first line, which is the document start
            new_content = _ensure_yaml_language_server_line(existing_content, schema_ref)
            if not new_content.endswith('\n'):
                new_content += '\n'
            new_content += '\n'.join(new_lines)
            dataset_path.write_text(new_content)

    @classmethod
    def _get_schema_path(cls, dataset_path: Path) -> Path:
        return dataset_path.parent / f'./{dataset_path.stem}_schema.json'

    @classmethod
    def _get_relative_path(cls, path: Path | str) -> Path:
        """Resolve relative paths as relative to the module in which the (sub)class is defined."""
        path = Path(path)

        if path.is_absolute():
            return path

        module_path = sys.modules[cls.__module__].__file__
        if module_path == __file__:
            raise ValueError(f'You should only call this method from a proper subclass of `{cls.__name__}`')

        assert module_path is not None, 'Module must be a file-based module'
        root = Path(module_path).parent
        return root / path


def _ensure_yaml_language_server_line(content: str, schema_ref: str) -> str:
    first_line = content.split('\n', 1)[0]
    yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
    if first_line == yaml_language_server_line:
        return content
    elif first_line.startswith('# yaml-language-server: $schema='):
        return '\n'.join([yaml_language_server_line] + content.split('\n')[1:])
    else:
        return f'{yaml_language_server_line}\n{content}'


async def _generate_examples(
    dataset_type: type[Dataset[Any, Any, Any]],
    path: Path,
    model: models.Model | models.KnownModelName = 'gpt-4o',
    n_examples: int = 3,
) -> list[DatasetRow[Any, Any, Any]]:
    if path.exists():
        cases_text = path.read_text()
        try:
            loaded = yaml.safe_load(cases_text)
        except yaml.YAMLError:
            raise ValueError(f'Cases file {path} is not valid YAML')

        try:
            existing_cases = dataset_type.model_validate(loaded).rows
        except ValidationError as e:
            raise ValueError(
                f'Cases file {path} contains data that does not match the schema. Delete the file before calling this function.'
            ) from e
    else:
        existing_cases = []

    n_examples = max(0, n_examples - len(existing_cases))
    if n_examples == 0:
        return []

    from pydantic_ai import Agent  # import locally to prevent circular dependencies

    agent = Agent(
        model,
        system_prompt=dedent('Generate concise example test cases that comply with the provided JSON schema.'),
        result_type=dataset_type,
        retries=1,
    )
    return (await agent.run(f'Generate {n_examples} examples')).data.rows


def _get_relative_path_reference(target: Path, source: Path, _prefix: str = '') -> Path:
    # Recursively resolve a relative path to target from source, adding '..' as needed.
    # This is useful for creating a relative path reference from a source file to a target file.
    # For example, if source is '/a/b/c.py' and target is '/a/d/e.py', the relative path reference
    # would be '../../d/e.py'.
    if not target.is_absolute():
        target = target.resolve()
    try:
        return Path(f'{_prefix}{Path(target).relative_to(source)}')
    except ValueError:
        return _get_relative_path_reference(target, source.parent, _prefix=f'{_prefix}../')


# TODO: Use this elsewhere in the library to make it easier to reuse
# def infer_name(value: Any, function_frame: types.FrameType | None) -> str | None:
#     """Infer the variable name from the call frame.
#
#     Usage should be `infer_name(value, inspect.currentframe())`.
#     """
#     if function_frame is not None and (parent_frame := function_frame.f_back):  # pragma: no branch
#         for name, item in parent_frame.f_locals.items():
#             if item is value:
#                 return name
#         if parent_frame.f_locals != parent_frame.f_globals:
#             # if we couldn't find the agent in locals and globals are a different dict, try globals
#             for name, item in parent_frame.f_globals.items():
#                 if item is value:
#                     return name
#     return None
