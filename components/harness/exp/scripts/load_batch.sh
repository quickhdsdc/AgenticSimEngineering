#!/usr/bin/env bash
set -uo pipefail

sm="urn:a4s:exp:sm:batch"

# Ensure submodel exists
aas sm tree "$sm" >/dev/null

# Count rows in CSV (excluding header)
rows=$(($(wc -l < ./batch.csv)-1))
echo "ROWS:$rows"

# Dry-run for the first row to confirm shape
first_name=$(awk -F, 'NR==2{print $1}' ./batch.csv)
first_value=$(awk -F, 'NR==2{print $2}' ./batch.csv)
echo "DRYRUN:${first_name}=${first_value}"
aas el add "$sm" '' --type Property --id-short "$first_name" --value "$first_value" --value-type xs:double --dry-run

# Create properties
count=0
{
  read -r header
  while IFS=, read -r name value; do
    if aas el add "$sm" '' --type Property --id-short "$name" --value "$value" --value-type xs:double; then
      count=$((count+1))
    else
      echo "ERROR creating $name" >&2
    fi
  done
} < ./batch.csv

echo "CREATED:$count"

# Validate submodel
if ! aas validate "$sm"; then
  echo "VALIDATION: issues reported" >&2
else
  echo "VALIDATION: ok"
fi
