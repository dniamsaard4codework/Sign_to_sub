# patches/

Out-of-tree patches that need to be applied to external dependencies after they're cloned/installed. See main project [README.md](../README.md) for context.

## `fairseq_signclip_windows.patch`

**Applies to:** [J22Melody/fairseq](https://github.com/J22Melody/fairseq) (the SignCLIP fork). See main README Step 5.

**Why needed:** vanilla upstream uses POSIX path conventions in 8 files. On Windows these crash at SignCLIP model load (`/` vs `\`, missing absolute paths, etc.). The patch is harmless on Linux/macOS — apply it on every platform for consistency.

**Generated against upstream commit:** `a8199440` (J22Melody/fairseq `main` branch, 2026-03-04 — "Merge pull request #16 from AmitMY/patch-1").

**Files modified (8 total, 36 lines changed):**

- `examples/MMPT/mmpt/models/mmfusion.py`
- `examples/MMPT/mmpt/processors/dsprocessor.py`
- `examples/MMPT/mmpt/processors/dsprocessor_sign.py`
- `examples/MMPT/mmpt/processors/processor.py`
- `examples/MMPT/mmpt/tasks/task.py`
- `examples/MMPT/mmpt/utils/load_config.py`
- `examples/MMPT/projects/retri/signclip_bsl/bobsl_islr_finetune_long_context.yaml`
- `examples/MMPT/scripts_bsl/extract_episode_features.py`

### How to apply

From the project root (after cloning fairseq into `fairseq_signclip/` per Step 5a):

```powershell
cd fairseq_signclip
git apply ..\patches\fairseq_signclip_windows.patch
cd ..
```

### How to verify it applied

```powershell
cd fairseq_signclip
git status                 # should list the 8 modified files
git diff --stat            # should show 8 files, ~36 lines changed
cd ..
```

### How to regenerate if upstream has moved on

If `git apply` fails because upstream files have changed since `a8199440`, regenerate from a working copy:

```powershell
# In your working fairseq_signclip with the manual patches applied:
cd fairseq_signclip
git diff --no-color > ..\patches\fairseq_signclip_windows.patch
cd ..
```

Then update the commit hash above. If conflicts are non-trivial, apply with `git apply --3way` for a merge-style resolution.
