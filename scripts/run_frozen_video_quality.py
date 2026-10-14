"""Enforce the frozen TEST gate, run fresh inference, then verify every frame."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path)
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--baseline-directory',type=Path,required=True)
    ap.add_argument('--baseline',default='b2634dd3430714c4adfad5a54b59030aa33cd248')
    args=ap.parse_args();root=args.directory
    frozen=json.loads((root/'frozen_selection.json').read_text(encoding='utf-8'))
    test=json.loads((root/'sign_TEST_frozen/report.json').read_text(encoding='utf-8'))
    if frozen['test_used_for_selection'] or frozen['selection_split']!='VAL' or test['split']!='TEST':
        raise RuntimeError('Invalid selection/test provenance')
    unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in frozen['runtime_sha256'].items())
    old=test['results']['base960_16'];new=test['results'][frozen['selected_method']];threshold=frozen['test_release_gate']
    checks={'runtime_unchanged_since_freeze':unchanged,
            'full_test_images':test['images']==2326,
            'recall_gain':new['code_recall']-old['code_recall']>=threshold['min_recall_gain'],
            'precision_drop_bounded':old['known_code_precision']-new['known_code_precision']<=threshold['max_precision_drop'],
            'large_sign_recall':new['ge32_recall']>=threshold['minimum_ge32_recall']}
    report={'passed':all(checks.values()),'checks':checks,'baseline':old,'candidate':new,
            'test_report_sha256':hashlib.sha256((root/'sign_TEST_frozen/report.json').read_bytes()).hexdigest(),
            'independent_field_video_accuracy':False}
    path=root/'test_release_gate.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');print('TEST_GATE',json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('Frozen candidate rejected; do not tune it on TEST')
    baseline_verified=json.loads((args.baseline_directory/'verification.json').read_text(encoding='utf-8'))
    if not baseline_verified['passed']:raise RuntimeError('Baseline is not verified')
    output=root/'full_v1'
    subprocess.run([sys.executable,str(ROOT/'scripts/complete_test_video.py'),str(args.source),
                    '--output',str(output),'--analysis-hz','3','--baseline',args.baseline,
                    '--baseline-predictions',str(args.baseline_directory/'predictions.jsonl'),
                    '--baseline-side','after'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_complete_video.py'),str(output)],cwd=ROOT,check=True)
    print('FULL_VIDEO_VERIFIED',str(output),flush=True)


if __name__=='__main__':main()
