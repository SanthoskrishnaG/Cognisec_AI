import os
import subprocess
import sys
from pathlib import Path

# Auto-resolve python binary to project .venv if current interpreter lacks requests
def _ensure_env():
    try:
        import requests
        return
    except ImportError:
        pass
    root = Path(__file__).resolve().parent.parent
    win_venv = root / ".venv" / "Scripts" / "python.exe"
    posix_venv = root / ".venv" / "bin" / "python"
    target = win_venv if win_venv.exists() else (posix_venv if posix_venv.exists() else None)
    if target and Path(sys.executable).resolve() != target.resolve():
        sys.exit(subprocess.call([str(target)] + sys.argv))

_ensure_env()
import requests

BASE = 'http://127.0.0.1:5000'

def run_tests():
    print('[1] Testing GET /')
    r = requests.get(f'{BASE}/')
    assert r.status_code == 200, f'Status {r.status_code}'
    assert 'Cognisec AI' in r.text
    assert 'view-overview' in r.text
    assert 'view-traffic-analysis' in r.text
    assert 'view-attack-detection' in r.text
    assert 'view-model-comparison' in r.text
    assert 'view-unseen-attacks' in r.text
    assert 'view-dataset-experiments' in r.text
    assert 'view-research-methodology' in r.text
    assert 'view-system-info' in r.text
    print('    [OK] Index HTML rendered with all 8 sections.')

    print('[2] Testing static assets')
    for asset in ['/css/styles.css', '/js/api.js', '/js/charts.js', '/js/app.js', '/figures/fig2_roc_pr_curves.png']:
        ra = requests.get(f'{BASE}{asset}')
        assert ra.status_code == 200, f'Asset {asset} failed with {ra.status_code}'
    print('    [OK] CSS, JS modules, and figures served.')

    print('[3] Testing API health')
    rh = requests.get(f'{BASE}/api/health')
    assert rh.status_code == 200
    h_data = rh.json()
    assert h_data['status'] == 'healthy'
    assert h_data['models']['xgboost']['loaded'] is True
    assert h_data['models']['vae']['loaded'] is True
    assert h_data['models']['preprocessor']['feature_count'] == 70
    xgb_thresh = h_data['models']['xgboost']['threshold']
    vae_thresh = h_data['models']['vae']['threshold']
    print(f'    [OK] System health: {h_data["status"]}, XGBoost threshold: {xgb_thresh}, VAE threshold: {vae_thresh}')

    print('[4] Testing API sample-csv')
    rs = requests.get(f'{BASE}/api/sample-csv')
    assert rs.status_code == 200
    s_data = rs.json()
    assert s_data['rows_count'] == 100
    csv_text = s_data['csv_text']
    print(f'    [OK] Sample CSV fetched ({s_data["rows_count"]} rows).')

    print('[5] Testing POST /api/analyze/csv-preview')
    rp = requests.post(f'{BASE}/api/analyze/csv-preview', json={'csv_text': csv_text})
    assert rp.status_code == 200
    p_data = rp.json()
    assert p_data['is_compatible'] is True
    assert p_data['matched_feature_count'] == 70
    assert len(p_data['preview_rows']) > 0
    print(f'    [OK] CSV Preview compatible: {p_data["matched_feature_count"]}/70 features matched.')

    print('[6] Testing POST /api/analyze/xgboost')
    rx = requests.post(f'{BASE}/api/analyze/xgboost', json={'csv_text': csv_text})
    assert rx.status_code == 200
    x_data = rx.json()
    assert x_data['total_flows'] == 100
    assert 'flagged_attacks' in x_data
    assert 'benign_flows' in x_data
    print(f'    [OK] XGBoost evaluated {x_data["total_flows"]} flows: {x_data["flagged_attacks"]} attacks, {x_data["benign_flows"]} benign.')

    print('[7] Testing POST /api/analyze/vae')
    rv = requests.post(f'{BASE}/api/analyze/vae', json={'csv_text': csv_text})
    assert rv.status_code == 200
    v_data = rv.json()
    assert v_data['total_flows'] == 100
    assert 'flagged_anomalies' in v_data
    assert 'normal_flows' in v_data
    print(f'    [OK] VAE evaluated {v_data["total_flows"]} flows: {v_data["flagged_anomalies"]} anomalies, {v_data["normal_flows"]} normal.')

    print('[8] Testing POST /api/analyze/dual')
    rd = requests.post(f'{BASE}/api/analyze/dual', json={'csv_text': csv_text})
    assert rd.status_code == 200
    d_data = rd.json()
    assert d_data['total_flows'] == 100
    summary = d_data['summary']
    print(f'    [OK] Dual Inference Summary:')
    print(f'         Both Flagged:  {summary["both_flagged"]} ({summary["both_flagged_pct"]}%)')
    print(f'         XGBoost Only:  {summary["xgboost_only"]} ({summary["xgboost_only_pct"]}%)')
    print(f'         VAE Only:      {summary["vae_only"]} ({summary["vae_only_pct"]}%)')
    print(f'         Both Normal:   {summary["both_normal"]} ({summary["both_normal_pct"]}%)')

    print('[9] Testing API comparison & metadata')
    rc = requests.get(f'{BASE}/api/results/comparison')
    assert rc.status_code == 200
    c_data = rc.json()
    assert len(c_data['known_attack_metrics']) == 6
    assert len(c_data['per_family_results']) == 8
    print('    [OK] Model comparison and disaggregation loaded.')

    print('\n>>> ALL LIVE HTTP SERVER ENDPOINTS AND INFERENCE PIPELINES VERIFIED 100% SUCCESSFUL! <<<')

if __name__ == '__main__':
    run_tests()
