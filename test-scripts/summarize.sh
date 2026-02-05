BASE_URL=http://localhost:8000

TEXT_CONTENT="This story is to track the development efforts required to implement the transport optimization requirement. Break down the tasks involved with a high-level estimate to build and test each and every criterion to achieve the optimization"

# Summarize a query
curl -v -XPOST "$BASE_URL/summarize" -d '{"user_input":"'$TEXT_CONTENT'"}' -H "Content-Type: application/json" | jq

