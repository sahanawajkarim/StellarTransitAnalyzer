
import os
import zipfile
import numpy as np
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

os.makedirs("lightcurves/confirmed", exist_ok=True)
os.makedirs("lightcurves/candidates", exist_ok=True)
os.makedirs("lightcurves/false_positives", exist_ok=True)

print("Fetching KOI table...")
koi_table = NasaExoplanetArchive.query_criteria(table="cumulative")

mask_confirmed = koi_table["koi_disposition"] == "CONFIRMED"
mask_candidates = koi_table["koi_disposition"] == "CANDIDATE"
mask_false = koi_table["koi_disposition"] == "FALSE POSITIVE"

confirmed_ids = koi_table["kepid"][mask_confirmed]
candidate_ids = koi_table["kepid"][mask_candidates]
false_ids = koi_table["kepid"][mask_false]

print(f"Confirmed KOIs: {len(confirmed_ids)}")
print(f"Candidate KOIs: {len(candidate_ids)}")
print(f"False Positive KOIs: {len(false_ids)}")


def download_lightkurves_lightkurve_resume(kepids, folder, zipname, limit=None):
    os.makedirs(folder, exist_ok=True)
    if limit:
        kepids = kepids[:limit]


    existing_files = set(os.listdir(folder))
    downloaded = 0

    for kid in kepids:
      
        quarters_done = {f for f in existing_files if f.startswith(f"KIC{kid}_") or f.startswith(f"KIC{kid}.fits")}
        if quarters_done:
            print(f"Skipping KIC {kid} (already has {len(quarters_done)} files)")
            continue

        try:
           
            search_result = lk.search_lightcurve(f"KIC {kid}", mission='Kepler')
            if len(search_result) == 0:
                print(f"No lightcurves found for KIC {kid}")
                continue

            lc_collection = search_result.download_all()

            if isinstance(lc_collection, lk.LightCurveCollection):
                for i, lc in enumerate(lc_collection):
                    lc_path = os.path.join(folder, f"KIC{kid}_Q{i+1}.fits")
                    if not os.path.exists(lc_path):  # Skip if already exists
                        lc.to_fits(lc_path)
                        existing_files.add(os.path.basename(lc_path))
            else:  # single LightCurve
                lc_path = os.path.join(folder, f"KIC{kid}.fits")
                if not os.path.exists(lc_path):
                    lc_collection.to_fits(lc_path)
                    existing_files.add(os.path.basename(lc_path))

            downloaded += 1
            print(f"Downloaded KIC {kid}")

        except Exception as e:
            print(f"Skipping KIC {kid} due to error: {e}")


    with zipfile.ZipFile(zipname, 'w') as zf:
        for root, _, files in os.walk(folder):
            for file in files:
                filepath = os.path.join(root, file)
                zf.write(filepath, os.path.relpath(filepath, folder))

    print(f"✅ Done: {zipname} ({downloaded}/{len(kepids)} new targets downloaded)")


download_lightkurves_lightkurve_resume(confirmed_ids, "lightcurves/confirmed", "Confirmed_Exoplanets.zip", limit=50)
download_lightkurves_lightkurve_resume(candidate_ids, "lightcurves/candidates", "Candidate_Exoplanets.zip", limit=None)
download_lightkurves_lightkurve_resume(false_ids, "lightcurves/false_positives", "FalsePositives.zip", limit=50)
