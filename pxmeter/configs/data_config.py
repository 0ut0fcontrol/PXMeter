# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import json
import logging
import os
from pathlib import Path

import gemmi
import requests

logging.basicConfig(level=logging.INFO)


def download_ccd_cif(output_path: Path):
    """
    Download the CCD CIF file from rcsb.org.

    Args:
        output_path (Path): The output path for saving the downloaded CCD CIF file.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading CCD CIF file from rcsb.org ...")

    output_cif = output_path / "components.cif"
    if output_cif.exists():
        logging.info("Remove old CCD CIF file: %s", output_cif)
        output_cif.unlink()

    url = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as f_in, output_cif.open("wb") as f_out:
            for chunk in iter(lambda: f_in.read(8192), b""):
                f_out.write(chunk)

    logging.info("Download CCD CIF file successfully: %s", output_cif)


def make_precomputed_json_from_ccd(
    components_file: Path, ccd_blocks_json: Path, one_letter_code_json: Path
):
    """
    Make a one-letter code JSON file from the CCD CIF file.

    Args:
        components_file (Path): The path to the CCD CIF file.
    """
    ccd_cif = gemmi.cif.read(str(components_file))

    ccd_code_to_one_letter_code = {}
    ccd_code_to_block = {}
    for block in ccd_cif:
        ccd_code = block.find_value("_chem_comp.id")
        one_letter_code = block.find_value("_chem_comp.one_letter_code")
        ccd_code_to_block[ccd_code] = block.as_string()
        if one_letter_code is None or one_letter_code == "?":
            continue
        ccd_code_to_one_letter_code[ccd_code] = one_letter_code

    with open(ccd_blocks_json, "w") as f:
        json.dump(ccd_code_to_block, f, indent=4)

    logging.info("Make CCD_BLOCKS_JSON successfully: %s", ccd_blocks_json)

    with open(one_letter_code_json, "w") as f:
        json.dump(ccd_code_to_one_letter_code, f, indent=4)
    logging.info("Make ONE_LETTER_CODE_JSON successfully: %s", one_letter_code_json)


# default is <repo_dir>/ccd_cache/components.cif Your path for components file
# You can change this path to your own path for components file by setting the environment variable:
# export PXM_CCD_FILE=<your_path>
repo_dir = Path(__file__).absolute().parent.parent.parent
ccd_file_in_repo = repo_dir / "ccd_cache" / "components.cif"
COMPONENTS_FILE = Path(os.environ.get("PXM_CCD_FILE", ccd_file_in_repo))

CCD_BLOCKS_JSON = COMPONENTS_FILE.with_suffix(".json")
ONE_LETTER_CODE_JSON = COMPONENTS_FILE.parent / "one_letter_code.json"

if not COMPONENTS_FILE.exists():
    logging.debug(
        "CCD CIF file not found. Downloading CCD CIF file to %s", COMPONENTS_FILE.parent
    )
    download_ccd_cif(output_path=COMPONENTS_FILE.parent)
    make_precomputed_json_from_ccd(
        COMPONENTS_FILE, CCD_BLOCKS_JSON, ONE_LETTER_CODE_JSON
    )
else:
    logging.debug("Load CCD CIF file from: %s", COMPONENTS_FILE)

if not ONE_LETTER_CODE_JSON.exists() or not CCD_BLOCKS_JSON.exists():
    make_precomputed_json_from_ccd(
        COMPONENTS_FILE, CCD_BLOCKS_JSON, ONE_LETTER_CODE_JSON
    )

logging.debug("Load CCD one-letter code from: %s", ONE_LETTER_CODE_JSON)
with open(ONE_LETTER_CODE_JSON, "r") as f:
    CCD_ONE_LETTER_CODE = json.load(f)


with open(CCD_BLOCKS_JSON, "r") as f:
    CCD_BLOCKS = json.load(f)
