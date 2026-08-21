import json
import pandas as pd
from pathlib import Path

result_files = sorted(Path("results").glob("sample*.results.json"))

if not result_files:
    raise FileNotFoundError("No sample JSON files found.")

print(f"Found {len(result_files)} sample files")

rows = []

for json_file in result_files:

    sample_id = json_file.name.replace(".results.json", "")

    print(f"Processing {sample_id}...")

    with open(json_file) as f:
        data = json.load(f)

    row = {
        "sample_id": sample_id,
        "main_lineage": data.get("main_lineage", ""),
        "sub_lineage": data.get("sub_lineage", ""),
        "drtype": data.get("drtype", ""),
        "total_dr_variants": len(data.get("dr_variants", []))
    }

    mutations = []
    drugs = []

    for variant in data.get("dr_variants", []):

        locus = variant.get("locus_tag", "")

        for drug_info in variant.get("drugs", []):

            drug = drug_info.get("drug", "")
            mutation = drug_info.get("original_mutation", "")

            if mutation:
                if locus:
                    mutation_id = f"{locus}_{mutation}"
                else:
                    mutation_id = mutation

                mutations.append(mutation_id)
                row[f"mutation_{mutation_id}"] = 1

            if drug:
                drugs.append(drug)
                row[f"drug_{drug}"] = 1

    mutations = sorted(set(mutations))
    drugs = sorted(set(drugs))

    row["mutation_count"] = len(mutations)
    row["mutations"] = ";".join(mutations)
    row["drugs_detected"] = ";".join(sorted(set(drugs)))

    rows.append(row)

df = pd.DataFrame(rows)

binary_columns = [
    c for c in df.columns
    if c.startswith("mutation_") or c.startswith("drug_")
]

if binary_columns:
    df[binary_columns] = df[binary_columns].fillna(0).astype(int)

output = Path("results/feature_table.csv")
df.to_csv(output, index=False)

print()
print("======================================")
print("FEATURE TABLE CREATED SUCCESSFULLY")
print("======================================")
print(f"Samples  : {len(df)}")
print(f"Features : {len(df.columns)}")
print(f"Output   : {output}")
print()

print(
    df[
        [
            "sample_id",
            "main_lineage",
            "sub_lineage",
            "drtype",
            "total_dr_variants",
            "mutation_count"
        ]
    ].to_string(index=False)
)
