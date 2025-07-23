# `pydantic_ai.tenacity`

## Setup

To use the tenacity integration, you need to install the `tenacity` dependency:

```bash
pip install tenacity
```

## Overview

The `pydantic_ai.tenacity` module provides HTTP transport wrappers and wait strategies that leverage the tenacity
library to add convenient retry behavior to HTTP requests. This is particularly useful for handling transient failures
like rate limits, network timeouts, or temporary server errors.

The module includes:
- **Transport Classes**: Wrap existing HTTP transports to add retry functionality
- **Wait Strategies**: Smart waiting that respects HTTP headers like `Retry-After`

For detailed usage examples and patterns, see the [HTTP Request Retries](../retries.md) guide.

::: pydantic_ai.tenacity
