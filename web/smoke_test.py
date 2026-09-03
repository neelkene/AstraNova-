import urllib.request, json, sys
sys.path.insert(0, 'd:/SIH')

base = 'http://127.0.0.1:5000'

def get(url):
    r = urllib.request.urlopen(url)
    return json.loads(r.read())

def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    r = urllib.request.urlopen(req)
    return json.loads(r.read())

tests_passed = 0
tests_failed = 0

def check(name, fn, *args, **kwargs):
    global tests_passed, tests_failed
    try:
        result = fn(*args, **kwargs)
        print(f"[PASS] {name}")
        tests_passed += 1
        return result
    except Exception as e:
        print(f"[FAIL] {name} -> {e}")
        tests_failed += 1
        return None

# GET endpoints
check("GET /api/health", get, f"{base}/api/health")
check("GET /api/test-components?limit=5", get, f"{base}/api/test-components?limit=5")
check("GET /api/component/SYN_C01216", get, f"{base}/api/component/SYN_C01216")
check("GET /api/model-performance", get, f"{base}/api/model-performance")

# POST screening
d24 = check("POST /api/screen/24h (SYN_C04946 Anomalous)", post, f"{base}/api/screen/24h", {"component_id": "SYN_C04946"})
if d24:
    print(f"      -> Decision={d24['decision']} Prob={d24['defect_probability']*100:.1f}% Drift={d24['predicted_168h_iddq_drift_pct']:.2f}%")

d96 = check("POST /api/screen/96h (SYN_C04946 Anomalous)", post, f"{base}/api/screen/96h", {"component_id": "SYN_C04946"})
if d96:
    print(f"      -> Decision={d96['decision']} Prob={d96['defect_probability']*100:.1f}% Drift={d96['predicted_168h_iddq_drift_pct']:.2f}%")

ds = check("POST /api/screen/sequential (SYN_C01216 Normal)", post, f"{base}/api/screen/sequential", {"component_id": "SYN_C01216"})
if ds:
    print(f"      -> FinalDecision={ds['final_decision']} at {ds['final_screening_gate']} EarlyExit={ds['early_exit_applied']}")

print(f"\n{'='*50}")
print(f"API Smoke Test: {tests_passed} passed / {tests_failed} failed")
