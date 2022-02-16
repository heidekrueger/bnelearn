# What does this Merge request do?

Describe in one or two sentences what your merge request does.
Tag any relevant gitlab issues.

## Checklist

Check all that apply. If your merge request does not comply with any of the points in the checklist, explain here.

- [ ] All gitlab issues closed or touched by this merge request have been mentioned in the request or the respective commits. I made sure only the correct ones are being closed by it.
- [ ] I have added/changed behaviour in the `bnelearn` module. I have read and applied the [Contribution Guidelines](/contributing.md))
    * [ ] I have added **no** user code in the module.
    * [ ] No absolute paths in the package, no setting of specific hardware (CPU, GPU, specific GPU) (Except in unit tests.)
    * [ ] I have written unit tests for any new or changed functionality in `bnelearn` package.
        * [ ] My tests will always pass if the implementation is working correctly.
        * [ ] My tests are written to test the functionality in the least reasonable amount of time.
        * [ ] Only descriptive `print()` statements in tests. (I.e. stuff that's useful if the test fails.)
    * [ ] I have used descriptive variable names everywhere.
    * [ ] I have used `pytorch` vectorization instead of python loops wherever reasonably possible or otherwise opened an issue in gitlab and added a `#TODO` to the source code referencing the issue number.
    * [ ] I have added documentation to any new or changed non-trivial function or class. (Minimum: docstring, inputs, outputs, type-hints)
    * [ ] I've run `pylint` on any changed files in the `bnelearn` package and cleaned up the warnings.
    * [ ] All tests pass locally.
- [ ] I have added/changed behaviour of experiment or analysis scripts or notebooks.
    * [ ] python scripts in `scripts/`
    * [ ] jupyter notebooks in `notebooks/`
    * [ ] R scripts or notebooks in `R/`
    * [ ] I have included information on where the script must be run (i.e. relative path to `bnelearn` module)
    * [ ] I have cleared the output of any jupyter before committing. I have **not** commited any Rmarkdown-notebook artefacts (i.e. generated html).
- [ ] I have added or updated package-wide documentation.
- [ ] I have added config files. (e.g. for git, gitlab, vscode, R, gitlab-runner, etc.)
    * [ ] I made sure the added config applies globally and no user- or machine-specific settings have been added.
- [ ] I have added artefacts such as data or figures
    This should only be done in exceptional cases. Please describe what and why.