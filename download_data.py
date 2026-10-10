"""Download the Inside Airbnb Vienna 2026-09-17 snapshot into data/."""

import argparse
import gzip
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


SNAPSHOT_DATE = "2026-09-17"
BASE_URL = f"https://data.insideairbnb.com/austria/vienna/vienna/{SNAPSHOT_DATE}"
DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = (
    ("data/listings.csv.gz", "listings.csv"),
    ("data/calendar.csv.gz", "calendar.csv"),
    ("data/reviews.csv.gz", "reviews.csv"),
    ("visualisations/neighbourhoods.csv", "neighbourhoods.csv"),
    ("visualisations/neighbourhoods.geojson", "neighbourhoods.geojson"),
)


def download_file(remote_path, filename, *, force=False):
    """Stream to temporary files; replace the destination only after success."""
    destination = DATA_DIR / filename
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"Skipping existing file: {destination}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{remote_path}"
    print(f"Downloading {url}", flush=True)
    # Temporary files live on the same filesystem for atomic replacement.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=DATA_DIR) as staging:
        archive = Path(staging) / "download"
        output = Path(staging) / filename
        request = Request(url, headers={"User-Agent": "ViennaStayAssistant/1.0"})
        with urlopen(request, timeout=120) as response, archive.open("wb") as target:
            expected_size = response.headers.get("Content-Length")
            shutil.copyfileobj(response, target)
        if expected_size is not None and archive.stat().st_size != int(expected_size):
            raise ValueError(f"Incomplete download for {filename}")

        if remote_path.endswith(".gz"):
            print(f"Decompressing {filename}", flush=True)
            with gzip.open(archive, "rb") as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
        elif remote_path.endswith(".zip"):
            # Copy only the expected member; never extract arbitrary ZIP paths.
            with zipfile.ZipFile(archive) as bundle:
                matches = [
                    entry for entry in bundle.infolist()
                    if not entry.is_dir() and Path(entry.filename).name == filename
                ]
                if len(matches) != 1:
                    raise ValueError(f"ZIP must contain exactly one {filename}")
                with bundle.open(matches[0]) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
        else:
            archive.replace(output)

        if output.stat().st_size == 0:
            raise ValueError(f"Downloaded file is empty: {filename}")
        output.replace(destination)
    print(f"Saved {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Download again and replace existing files (use to change snapshots).",
    )
    args = parser.parse_args()
    failures = []
    for remote_path, filename in FILES:
        try:
            download_file(remote_path, filename, force=args.force)
        except Exception as error:
            failures.append(filename)
            print(f"Failed to download {filename}: {error}", file=sys.stderr)
    if failures:
        print("Download incomplete. Run the script again to retry failed files.", file=sys.stderr)
        return 1
    print(f"All five data files are available in {DATA_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
