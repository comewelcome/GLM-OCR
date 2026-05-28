#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./test_ocr.sh [pdf-file]
# Example:
#   ./test_ocr.sh AIDATA_translated_eng.pdf

PDF_FILE=${1:-Plan.pdf}
OUTPUT_FILE=${OUTPUT_FILE:-/tmp/ocr_response.json}

if [ ! -f "$PDF_FILE" ]; then
  echo "Fichier introuvable: $PDF_FILE" >&2
  exit 1
fi

# Resolve API port from Docker Compose if available, fallback to 3001
API_PORT=${API_PORT:-}
if [ -z "$API_PORT" ] && command -v docker >/dev/null 2>&1 && [ -f docker-compose.yml ]; then
  API_PORT=$(docker compose port api 3000 2>/dev/null | sed -n 's/.*:\([0-9]\+\)$/\1/p' || true)
fi
API_PORT=${API_PORT:-3001}

API_URL="http://localhost:${API_PORT}/v1/ocr"

echo "Using API URL: $API_URL"

echo "Uploading $PDF_FILE..."
curl -s -X POST "$API_URL" -F "file=@${PDF_FILE}" -o "$OUTPUT_FILE"

if [ ! -s "$OUTPUT_FILE" ]; then
  echo "Aucune réponse reçue; vérifiez si l'API est démarrée sur le port $API_PORT" >&2
  exit 1
fi

echo "Response saved to $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"
echo '--- response head ---'
head -n 50 "$OUTPUT_FILE"

echo '--- parse response ---'
python3 - <<PY
import json
from pathlib import Path
path = Path(r"$OUTPUT_FILE")
text = path.read_text()
try:
    data = json.loads(text)
except json.JSONDecodeError as exc:
    print('JSON parse error:', exc)
    print(text[:1000])
    raise
print('id=', data.get('id'))
print('pages=', len(data.get('pages', [])))
if data.get('pages'):
    page1 = data['pages'][0]
    print('page1 markdown preview:')
    print(page1.get('markdown', '')[:1000].replace('\n', '\\n'))
PY
