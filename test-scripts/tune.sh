BASE_URL=http://localhost:8000

# Tune the model with a sample one-shot prompt
payload='{"input":"'$1'","output":"'$2'"}'
echo $payload
curl -v -XPOST "$BASE_URL/tune" -d "$payload" -H "Content-Type: application/json" | jq

