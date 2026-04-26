# Code Review: sdt_read

## TODO List

Fix these one at a time — tag Claude on each when ready.

- [ ] **1. BUG: Time calculation formula in `read_bruker_sdt.py:69`**
  Line 69 computes `(tac_r / tac_g) * adc_re`, but all other readers and the commented-out original use `tac_r / (tac_g * adc_re)`. One of these is wrong.

- [ ] **2. BUG: `dataspl = None` crash path in `read_bruker_sdt.py:111-113`**
  If the generic `except Exception` branch is hit, `dataspl` is set to `None`, and the next line `np.fromstring(None, ...)` will crash. Need to raise or return early.

- [ ] **3. Replace deprecated `np.fromstring` with `np.frombuffer`**
  Affected files: `read_bruker_sdt.py:115`, `read_wiscscan_sdt.py:66`.

- [ ] **4. Bare `except` in `read_wiscscan_sdt.py:28`**
  Catches all exceptions including `KeyboardInterrupt`. Narrow to `except (KeyError, IndexError):`.

- [ ] **5. Inconsistent `record_dtype` vs `np.dtype(MEASURE_INFO)` across readers**
  `read_bruker_sdt.py` uses the newer `record_dtype()` approach. `read_openscan_sdt.py` and `read_wiscscan_sdt.py` use the old `np.dtype()` approach which may break with newer `sdtfile` versions. Unify them.

- [ ] **6. Inconsistent header field name checks**
  `read_bruker_sdt.py` checks for both `meas_desc_block_offset` and `meas_desc_block_offs`. `read_openscan_sdt.py` accesses `header.setup_offs` directly without fallback. Make all readers handle both naming conventions.

- [ ] **7. File opened multiple times in `read_bruker_sdt.py`**
  Same file opened 3 times (lines 15, 27, 52). Refactor to open once and seek between sections.

- [ ] **8. `os.chdir` side effect in `openscanSdt_toTiff.py:11`**
  Mutates global working directory. Replace with `glob.glob(os.path.join(folder, '*.sdt'))` and use full paths.

- [ ] **9. Hardcoded test path in `test.py:10`**
  `C:\dev\test_data` only works on one machine. Use an env var or skip decorator.

- [ ] **10. Remove unused imports**
  - `read_bruker_sdt.py`: `SdtFile` (line 6), `pathlib` (line 4), `io` (line 5)
  - `test.py`: `os` (line 2)

- [ ] **11. Add `__init__.py` for package structure**
  No `__init__.py` exists, making this hard to import as a package. Add one exposing the public API (`read_bruker_sdt`, `read_sdt_openscan`, `read_sdt_wiscscan`).
