#!/bin/bash
echo "🚀 Starting Full Verification Pipeline..."
source venv/bin/activate

# 1. Start API Server (Background)
# We use port that isn't seemingly busy, say 8000 (user's port)
echo "🌐 Starting API Server..."
python -m uvicorn api:app --host 0.0.0.0 --port 8009 > api.log 2>&1 &
API_PID=$!
echo "   PID: $API_PID"
sleep 5

# 2. Run Test Script against API
echo "🧪 Running API Tests (Search, Process, Error Handling)..."
python mock_api_test.py
TEST_EXIT=$?

if [ $TEST_EXIT -ne 0 ]; then
    echo "❌ API Tests Failed!"
    kill $API_PID
    exit 1
fi

# 3. Verify Batch Worker
echo "🏗️ Running Batch Worker on 'tests/test_media'..."
python bo_worker.py --dir tests/test_media
WORKER_EXIT=$?

if [ $WORKER_EXIT -eq 0 ]; then
    echo "✅ Batch Worker Success"
else
    echo "❌ Batch Worker Failed"
fi

# Cleanup
echo "👋 Stopping API Server..."
kill $API_PID
echo "✅ Done."
