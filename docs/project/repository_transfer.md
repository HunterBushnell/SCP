# Repository Transfer Checklist

This checklist accompanies the prepared **local, unpublished v0.3.0
checkpoint** for transferring `HunterBushnell/SCP` to `cyneuro/SCP`.
Version metadata and [release notes](release_notes_v0.3.0.md) describe completed
workflow improvements; they do not perform the GitHub transfer, change remotes,
commit, push, tag, or create a release. Canonical repository, issue, citation,
and notebook links are prepared for the new owner; their live behavior has not
yet been verified after transfer.

SCP is a project of the Neural Engineering Laboratory at the University of
Missouri (Mizzou), [CyNeuro](https://github.com/cyneuro). Hunter Bushnell remains
the original creator and SCP contact: [hrbncv@missouri.edu](mailto:hrbncv@missouri.edu).
The citation author and original affiliation are preserved.

## 1. Publish the reviewed checkpoint

- [ ] Review the local patch and preserve unrelated work. This ownership update
  is not scientific cleanup: leave A1, production/private workspaces, model
  assets, data, and results alone. `LICENSE` and `THIRD_PARTY_NOTICES.md` must
  remain byte-identical.
- [ ] Review the intended file set, v0.3.0 metadata/release notes, and local
  validation results. Do not stage ignored research/private files.
- [ ] Commit and push the reviewed checkpoint to the existing repository using
  the lab's normal review process. This guide does not run those commands.
  The new canonical links may not resolve until the transfer is complete.
  A version update and commit do not automatically create a Git tag or release.

## 2. Transfer manually on GitHub

- [ ] Confirm administrator access to the source and permission to create
  repositories in CyNeuro. Check that the destination does not already contain
  `SCP` or a conflicting fork; do not create a replacement repository.
- [ ] In the source repository's **Settings → Danger Zone → Transfer**, choose
  `cyneuro` and retain the name `SCP` and current visibility. The user performs
  this step manually; follow GitHub's [repository transfer instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository).

## 3. Verify ownership, access, settings, and apps

- [ ] Open [cyneuro/SCP](https://github.com/cyneuro/SCP) and confirm the intended
  owner, visibility, default branch, history, issues/PRs, tags, and releases.
- [ ] Verify Hunter's access and intended lab/team permissions. Review the
  organization defaults, branch protection/rulesets, Actions permissions,
  secrets/environments, webhooks, deploy keys, and Pages settings if used.
- [ ] Check connected GitHub Apps and integrations, including their organization
  approval and repository selection. Reauthorize only where needed; do not
  assume the old installation or grant covers the new owner.

## 4. Update the local remote

Only after the transfer and access checks succeed, the user may update this
checkout's `origin` (preserving HTTPS or SSH as appropriate):

```bash
git remote -v
git remote set-url origin https://github.com/cyneuro/SCP.git
git remote -v
```

- [ ] Update other local clones when next used. The existing commit history
  transfers with the repository; the checkpoint need not be pushed again.

## 5. Check the published links and notebook entry points

- [ ] Open the canonical repository, [issues](https://github.com/cyneuro/SCP/issues),
  documentation links, and citation metadata from the published branch.
- [ ] Open all nine public Colab links (the eight root notebooks and
  `extra_notebooks/act_segmentation.ipynb`) against the published branch and
  verify that they resolve to `cyneuro/SCP`. In a fresh Colab runtime, check the
  bootstrap/clone path and a representative core workflow before claiming
  Colab execution validation. Link resolution alone is not an execution test.
- [ ] From a project directory outside the SCP checkout, activate the normal
  SCP environment and check `python -c "import scp_pipeline"` and
  `python -m scp_pipeline --help`. A repository-owner change does not require
  reinstalling an editable checkout that remains at the same local path, but
  the v0.3.0 metadata update does: run
  `python -m pip install -e /path/to/SCP` in that environment. This also applies
  to a newly cloned or relocated checkout. Restart notebook kernels after code
  updates; see the [dependency workflow](../guides/dependency_workflow.md).
- [ ] Check the old repository URL's redirect. Do **not** recreate
  `HunterBushnell/SCP` as a repository or fork: that can permanently remove
  GitHub's transfer redirect. Pages URLs require separate checking, because
  Pages sites are not covered by repository redirects. See
  [GitHub's redirect guidance](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository#whats-transferred-with-a-repository).
- [ ] Record the actual post-transfer results and update this checklist's
  preparation status when publication and verification are complete.

The user confirmed setup discovery, save, and copy feedback working after a
VS Code window reload. That limited confirmation does not establish broad
browser coverage, complete ACT optimization, or approve the paused/default-off
synaptic-recording work. Those remain separate [roadmap](roadmap.md) items.
