# Contribution Process
All contributions of source code should fit the following pattern.

1. Make or pick an issue.
1. Create a merge-request from the issue page. Edit the merge-request to use our default template and describe list of changes you wish to commit.
1. Implement your changes. **Commit only to the merge-request topic branch** created above. Use commit message tags such as `Fixed xyz. closes #<issue no>`.
1. When you think your implementation is done, go through the checklist in the merge-request description and make sure you comply with the contribution guidelines above. If you want or need to violate a point on the checklist, start a discussion in the merge request and tag a maintainer (@heidekrueger for now).
1. Gitlab is set to only allow semi-linear git histories. If your merge request is based on an outdated version of `master`, you might have to rebase. (Gitlab interface should help you to take care of this.)
1. When you're done, resolve the WIP status of the merge request, and assign the request to a maintainer who will review your request.
1. Implement any changes the maintainer asks you to do, then reassign.

# Contribution Guidelines

* The `bnelearn` subdirectory is a python module that should be thought of like a package.
  No user code (scripts running experiments etc.) should go in here, with the possible exception of tutorials/examples.
* Everything in `bnelearn` should
    * be written with performance, reliability and maintainability in mind, e.g. 
        * Use descriptive variable names
        * Write code as explicitly as possible (unless this has grave performance impact)
        * Validate inputs to functions, handle edge cases and errors (i.e. division by 0). Where validation has a major performance impact, add an option to disable it.
        * Use native `pytorch` operations over loops whenever possible. If you need a quick working copy, add an `Optimization` issue and TODO as decribed below.
    * well documented.  At a minimum, this means function signatures with type hints.
    * covered by **Unit Tests** in `bnelearn/tests`.
    * Adhere to PEP-8 python style guide (lint using `pylint`). This also applies to the tests! Some exceptions are outlined in `.pylintrc`. If you deliberately make the choice to violate pep8, add a `#pylint: disable=rule-that-you-broke` to the relevant codeblock to suppress linter warnings.
    * If you know that some code should be changed in the future, make a gitlab issue (add low-priority tag if appropriate) and add a `#TODO:` comment in the source with a one-line description and reference to the gitlab issue.
* User code goes in `scripts` for python scripts, `notebooks` for jupyter notebooks and  `R` for R scripts and Rmarkdown files (for analysis, generating figures, etc).
    * User code does not have to adhere to the standards above. 
    * Try to keep your user code modular and readable, think about whether you can turn parts of it into core package functionality later on.

# Taking Snapshots of the Repository
When submitting a paper, releasing code etc, we need to create a persistent snapshot of the repo. 
This should be done in the following way:

1. At the state of the project, create a branch named for the snapthot from master, i.e. `WITS-submission-2019`.
1. In this branch, add any artefacts that are not checked into master, but should be in the snapshot such as
    * Analysis code
    * Experiment Data
    * Figures
    * Documents

    If your total data is larger than single-digit megabytes, use Git Large File Storage. To do so, first install `git-lfs` on your machine (Ubuntu: `apt-get install git-lfs`, Windows: https://git-lfs.github.com/).
    You should put all your artefact files into a single archive to save space and bandwith, e.g.

    ```tar -czvf experiments/SNAPSHOT_NAME_EXPERIMENTS.tar.gz experiments/directory-or-file```
    **For safety, do so in your experiments folder, which is by default ignored by git.**

    Once you have created the archive and installed git-lfs on your *system*, you need to also activate it in the repository and tell it to track your archive files.

    ```
    git lfs install                       # initialize the Git LFS project
    git lfs track "*.tar.gz"              # select the file extensions that you want to treat as large files
    git add .gitattributes                # modified by git lfs track. must be added to make sure git handles lfs files correctly!
    git add -f experiments/SNAPSHOT_archive.tar.gz # add your archive. -f because experiments directory is usually ignored.
    ```

1. To make the snapshot reproducible, we need the full status of the conda env includign package version numbers. Export the current conda env using `conda env export` and add the results to a file `conda-env-export.yml` in the root of the repository.
1. Add a description of the snapshot including the current date in the top of readme.md.
1. Create a "Tag" on the branch in gitlab.
1. Go to gitlab Settings -> repository -> protected tags and protect the tag (can edit: no one)