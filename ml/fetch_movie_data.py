"""Download the IMDb genre-classification corpus into `data/`.

The two `.txt` files are ~35 MB each and are deliberately not committed. The
original `Movie_genere_classification.py` expected them to be present locally;
this script makes that dependency reproducible.
"""

from __future__ import annotations

import os
import urllib.request

import artifact as art

BASE_URL = (
    "https://raw.githubusercontent.com/GojoUchiha/codsoft_task1/master/"
    "Genre%20Classification%20Dataset/"
)
FILES = ["train_data.txt", "test_data_solution.txt"]


def main() -> None:
    target_dir = os.path.join(art.REPO_ROOT, "data")
    os.makedirs(target_dir, exist_ok=True)
    for name in FILES:
        destination = os.path.join(target_dir, name)
        if os.path.exists(destination) and os.path.getsize(destination) > 1_000_000:
            print(f"  {name} already present, skipping")
            continue
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, destination)
        print(f"    {os.path.getsize(destination) / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
