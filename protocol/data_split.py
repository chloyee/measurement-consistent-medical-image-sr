"""Assign patient-disjoint train, validation and test partitions."""
import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with patient_id,image_path and optional series_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-fraction", type=float, default=.15)
    parser.add_argument("--test-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or not 0 < args.test_fraction < 1 or args.val_fraction+args.test_fraction >= 1:
        parser.error("Fractions must be positive and sum to less than one")
    source, target = Path(args.input).resolve(), Path(args.output).resolve()
    if source == target or target.exists():
        parser.error("Choose a new output manifest path")
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"patient_id", "image_path"} <= set(reader.fieldnames or []):
            parser.error("Input needs patient_id and image_path")
        rows = list(reader)
    patients = sorted({r["patient_id"].strip() for r in rows})
    if not all(patients) or len(patients) < 3:
        parser.error("At least three nonempty patient IDs are required")
    random.Random(args.seed).shuffle(patients)
    nv, nt = max(1,round(len(patients)*args.val_fraction)), max(1,round(len(patients)*args.test_fraction))
    if nv+nt >= len(patients):
        parser.error("Fractions leave no training patients")
    parts = {p:("val" if i<nv else "test" if i<nv+nt else "train") for i,p in enumerate(patients)}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["patient_id","series_id","partition","image_path"])
        writer.writeheader()
        for r in rows:
            image=Path(r["image_path"])
            image=(source.parent/image).resolve() if not image.is_absolute() else image
            p=r["patient_id"].strip()
            writer.writerow({"patient_id":p,"series_id":r.get("series_id") or p,
                             "partition":parts[p],"image_path":str(image)})


if __name__ == "__main__":
    main()
