# `pxm gen-input` Usage Guide

`pxm gen-input` converts structural inputs across multiple formats—**mmCIF → AF3 / Protenix / Boltz**, **AF3 ←→ Protenix**, etc.

---

## 🧭 Command Overview

```bash
pxm gen-input \
  -i INPUT_PATH \
  -o OUTPUT_PATH \
  -it cif|af3|protenix|boltz \
  -ot af3|protenix|boltz \
  [--seeds "0,1,2" | --num-seeds 5] \
  [--assembly-id 1] \
  [--num-cpu 8]
```

Supported input types:

* `cif` - mmCIF structure
* `af3` - AlphaFold3 JSON
* `protenix` - Protenix JSON
* `boltz` - Boltz YAML

Supported output types:

* `af3`, `protenix`, `boltz`

The tool works on **single files** or **directories** (flat directory only).

---

## 🎮 Interactive Mode

If you don't have a source file and want to build a model input from scratch, you can use the **Interactive Mode**.

### How to Start
```bash
pxm gen-input -I
# or
pxm gen-input --interactive
```

### Features
*   **Step-by-step Guidance**: The tool will walk you through selecting the output format, naming the job, and adding components.
*   **Load from Existing File**: You can optionally initialize your complex by loading components and bonds from an existing file (`.cif`, `.json` for AF3/Protenix, or `.yaml` for Boltz).
*   **Component Management**:
    *   **Add Polymer**: Enter sequence strings (validated against standard alphabets) and add modifications at specific positions.
    *   **Add Ligand**: Support for CCD codes, SMILES, and file paths (validated against model-specific limits).
    *   **Remove Component**: Easily remove any added chain. All affected covalent bonds will be automatically cleaned up or re-indexed.
*   **Covalent Bonds**: Add bonds between any two atoms across chains with real-time range validation for Residue IDs.
*   **User-friendly Interface**:
    *   **Numbered Menus**: Quick selection using numbers (1, 2, 3...) instead of typing commands.
    *   **Smart Defaults**: Press `Enter` to accept recommended values (marked with `*`).
    *   **Live Preview**: See your complex grow as you add or modify components.

---

## ⚙️ Key Arguments

### Mandatory

| Flag                 | Description              |
| -------------------- | ------------------------ |
| `-i, --input`        | Input file or directory  |
| `-o, --output`       | Output file or directory |
| `-it, --input-type`  | Input format             |
| `-ot, --output-type` | Output format            |
| `-p, --pdb-ids`      | Filter inputs by PDB IDs (comma-separated or file path) |

**Input and output formats must differ.**
**File-to-file or dir-to-dir only.**

---

### Seeds (for AF3 / Protenix output only)

You must provide exactly one of:

* `--seeds "0,1,2"` — explicit list
* `--num-seeds N` — generates seeds `[0…N-1]`

Boltz output **does not use seeds**.

---

### CIF-specific options

| Flag            | Description                      |
| --------------- | -------------------------------- |
| `--assembly-id` | Biological assembly ID to expand |

---

### Parallelism

`--num-cpu N`
Number of workers (Joblib). `-1` uses all available CPUs.

---

## 🐍 Python API

You can call the same logic from Python instead of the CLI.

### High-level entry point

The CLI `pxm gen-input` is a thin wrapper around `run_gen_input`:

```python
from pathlib import Path
from pxmeter.input_builder.gen_input import run_gen_input

run_gen_input(
    input_path=Path("./cifs"),
    output_path=Path("./af3_inputs"),
    input_type="cif",
    output_type="af3",
    seeds=None,          # use num_seeds OR seeds, not both
    num_seeds=5,
    assembly_id="1",
    num_cpu=8,
)
```

Rules are the same as the CLI:

* `input_type` / `output_type` must differ.
* For `output_type` in `{ "af3", "protenix" }` you must provide **either** `seeds` **or** `num_seeds`.
* For `output_type == "boltz"`, both `seeds` and `num_seeds` can be left as `None`.

Example: Protenix → Boltz (no seeds needed):

```python
from pathlib import Path
from pxmeter.input_builder.gen_input import run_gen_input

run_gen_input(
    input_path=Path("protenix.json"),
    output_path=Path("boltz.yaml"),
    input_type="protenix",
    output_type="boltz",
    # seeds / num_seeds not required for Boltz
)
```

### Lower-level helpers

If you already have explicit file mappings, you can use the lower-level helpers:

```python
from pathlib import Path
from pxmeter.input_builder.gen_input import gen_one, gen_batch

# Single file
gen_one(
    input_f=Path("structure.cif"),
    output_f=Path("af3.json"),
    input_type="cif",
    output_type="af3",
    seeds=[0, 1, 2],
    assembly_id="1",
)

# Batch (list of (input, output) pairs)
pairs = [
    (Path("cifs/1abc.cif"), Path("af3/1abc.json")),
    (Path("cifs/2xyz.cif"), Path("af3/2xyz.json")),
]

gen_batch(
    input_and_output_files=pairs,
    input_type="cif",
    output_type="af3",
    seeds=[0, 1, 2],
    assembly_id="1",
    num_cpu=8,
)
```

These functions do not infer file lists or suffixes; they only perform the conversion.

---

## 📝 Usage Examples

### Batch mmCIF → AF3

```bash
pxm gen-input \
  -i ./cifs \
  -o ./af3_inputs \
  -it cif -ot af3 \
  --num-seeds 5 \
  --assembly-id 1 \
  --num-cpu 8
```

### AF3 → Protenix

```bash
pxm gen-input \
  -i af3.json \
  -o protenix.json \
  -it af3 -ot protenix \
  --seeds "0"
```

### Protenix → Boltz

```bash
pxm gen-input \
  -i protenix.json \
  -o boltz.yaml \
  -it protenix -ot boltz
```

### mmCIF → Boltz

```bash
pxm gen-input \
  -i structure.cif \
  -o boltz.yaml \
  -it cif -ot boltz
```
t af3 \
  --num-seeds 5 \
  --assembly-id 1 \
  --num-cpu 8
```

### AF3 → Protenix

```bash
pxm gen-input \
  -i af3.json \
  -o protenix.json \
  -it af3 -ot protenix \
  --seeds "0"
```

### Protenix → Boltz

```bash
pxm gen-input \
  -i protenix.json \
  -o boltz.yaml \
  -it protenix -ot boltz
```

### mmCIF → Boltz

```bash
pxm gen-input \
  -i structure.cif \
  -o boltz.yaml \
  -it cif -ot boltz
```
