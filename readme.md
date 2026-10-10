# Vienna Stay Assistant

Group project for answering questions about Vienna short-term accommodation using structured listing data and guest reviews.

## Download the data

The datasets are excluded from Git. After cloning this repository, **run the download script before using the data**:


The destination is relative to the script, so it works regardless of your terminal's current directory. Downloads and decompression stream to disk to avoid loading large datasets into memory. Allow disk space for the extracted CSVs and temporary compressed files.

Existing non-empty files are skipped. Their snapshot date is not checked. If you already have data from another snapshot, replace it explicitly:

```sh
python download_data.py --force
```

Successful downloads are kept if another file fails. The script reports failures and exits with a nonzero status; run it again to retry missing files. During replacement, an existing file is preserved until its download and extraction succeed. Temporary archives are removed after processing.

The supplied archives use gzip (`.gz`), rather than ZIP. The script also supports a `.zip` URL containing a single matching output filename if a source is changed later.

## Data source and attribution

Data source: [Inside Airbnb](https://insideairbnb.com/), Vienna, Austria, snapshot 2026-09-17. Data is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

- [Detailed listings](https://data.insideairbnb.com/austria/vienna/vienna/2026-09-17/data/listings.csv.gz)
- [Calendar](https://data.insideairbnb.com/austria/vienna/vienna/2026-09-17/data/calendar.csv.gz)
- [Detailed reviews](https://data.insideairbnb.com/austria/vienna/vienna/2026-09-17/data/reviews.csv.gz)
- [Neighbourhoods CSV](https://data.insideairbnb.com/austria/vienna/vienna/2026-09-17/visualisations/neighbourhoods.csv)
- [Neighbourhood boundaries](https://data.insideairbnb.com/austria/vienna/vienna/2026-09-17/visualisations/neighbourhoods.geojson)