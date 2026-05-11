import json

with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
    data = json.load(f, strict=False)

unique_keys = set()
for item in data:
    for k in item.get("keys", []):
        unique_keys.add(k)

print("Unique keys:")
for k in unique_keys:
    print(f"- {k}")
