---
name: Bug report
about: Report a reproducible defect in the framework
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the bug
A clear description of what the bug is.

## To Reproduce
Steps to reproduce (include the exact function call and inputs):

```python
from src.XXX import yyy
yyy(...)
```

## Expected behavior
What you expected to happen.

## Actual behavior
What actually happens (include the full traceback).

## Environment
- Python version:
- OS:
- `pip freeze` output (relevant packages):

## Regression test
Does a `@pytest.mark.regression` test cover this? If not, add one as part of the fix.
