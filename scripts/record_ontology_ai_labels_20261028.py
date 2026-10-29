"""Record per-sheet AI labels for out-of-ontology boxes (pass A / pass B), then merge by agreement.

record : --pass A --sheet 3 "1=P.105 2=N 3=U ..."   N = not_qcvn_sign, U = uncertain, otherwise a QCVN code.
status : sheets recorded per pass.
merge  : pass A x pass B -> data/manifests/dataset_v2/ai_ontology_extension_labels.jsonl (no image bytes).
Pass B is recorded without reading pass A's file; `merge` is the first command that opens both.
"""
from __future__ import annotations
import argparse,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'output/kiet_wp_20261028/recognizer_v2'
MANIFEST=ROOT/'data/manifests/dataset_v2/ai_ontology_extension_labels.jsonl'
NOT_QCVN='not_qcvn_sign';UNCERTAIN='uncertain'
REVIEWER='ai-review (Claude Code)'
V12_CLASSES=ROOT/'data/audit/semantic/final_v1_2/recognition_classes_v1_2.csv'


def valid_codes():
    index=json.loads((OUT/'qcvn_refs/index.json').read_text(encoding='utf-8'))
    codes=set(index['codes'])
    import csv
    with V12_CLASSES.open(encoding='utf-8',newline='') as stream:
        codes|={row['canonical_sign_code'] for row in csv.DictReader(stream)}
    return codes,index


def clean_name(name):
    # The PDF text layer splits a few Vietnamese words ('tr ái').
    return name.replace('tr ái','trái') if name else name


def sheet_manifest(name):
    return json.loads((OUT/f'sheets_{name}/manifest.json').read_text(encoding='utf-8'))


def label_path(name):return OUT/f'labels_{name}.jsonl'


def read_labels(name):
    path=label_path(name)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.is_file() else []


def parse_assignments(text,expected,codes):
    labels={}
    for token in text.split():
        idx,_,value=token.partition('=')
        if not idx.isdigit() or not value:raise ValueError(f'Bad token {token!r}')
        idx=int(idx)
        if idx in labels:raise ValueError(f'Duplicate index {idx}')
        if value=='N':labels[idx]=NOT_QCVN
        elif value=='U':labels[idx]=UNCERTAIN
        elif value in codes:labels[idx]=value
        else:raise ValueError(f'Unknown code {value!r} at {idx}')
    if set(labels)!=set(range(1,expected+1)):
        raise ValueError(f'Missing {sorted(set(range(1,expected+1))-set(labels))} extra {sorted(set(labels)-set(range(1,expected+1)))}')
    return labels


def record(args):
    codes,_=valid_codes();manifest=sheet_manifest(args.pass_name)
    sheet=next(s for s in manifest['sheets'] if s['sheet']==args.sheet);done={r['sheet'] for r in read_labels(args.pass_name)}
    if args.sheet in done and not args.force:raise SystemExit(f'Sheet {args.sheet} already recorded for pass {args.pass_name}')
    labels=parse_assignments(args.assignments,len(sheet['entries']),codes)
    rows=[r for r in read_labels(args.pass_name) if r['sheet']!=args.sheet]
    rows+=[dict(sheet=args.sheet,idx=e['idx'],canonical_box_id=e['canonical_box_id'],split=e['split'],label=labels[e['idx']]) for e in sheet['entries']]
    label_path(args.pass_name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=lambda r:(r['sheet'],r['idx']))),encoding='utf-8')
    counts=collections.Counter(labels.values());print(f'pass {args.pass_name} sheet {args.sheet}: recorded {len(labels)}; N={counts[NOT_QCVN]} U={counts[UNCERTAIN]} coded={len(labels)-counts[NOT_QCVN]-counts[UNCERTAIN]}')


def status(args):
    for name in ('A','B'):
        total=len(sheet_manifest(name)['sheets']);done=sorted({r['sheet'] for r in read_labels(name)})
        print(f'pass {name}: {len(done)}/{total} sheets; next={next((s for s in range(1,total+1) if s not in done),None)}')


def merge(args):
    _,index=valid_codes();a={r['canonical_box_id']:r for r in read_labels('A')};b={r['canonical_box_id']:r for r in read_labels('B')}
    if len(a)!=1531 or len(b)!=1531:raise SystemExit(f'Incomplete passes: A={len(a)} B={len(b)}')
    rows=[];stats=collections.defaultdict(collections.Counter)
    for box_id in sorted(a):
        la,lb=a[box_id]['label'],b[box_id]['label'];split=a[box_id]['split']
        is_code=lambda x:x not in (NOT_QCVN,UNCERTAIN)
        agreed=is_code(la) and la==lb
        code=la if agreed else None;heading=index['headings'].get(code) if code else None
        entry=index['codes'].get(code) if code else None
        ref=None
        if code:
            figures=entry['figure_ids'] if entry else []
            ref=dict(figures=['Hình '+f for f in figures],appendix_item=(heading or {}).get('item'),pdf_page=(heading or {}).get('page'),
                name_scope=('group_heading' if (heading or {}).get('group_heading') else 'mention' if (heading or {}).get('mention_only') else 'code_heading'),
                source='QCVN 41:2024/BGTVT')
        rows.append(dict(canonical_box_id=box_id,split=split,code=code,name_vi=clean_name((heading or {}).get('name_vi')) if code else None,
            pass_a=la,pass_b=lb,agreed=agreed,reviewer=REVIEWER,provenance='ai_reviewed',qcvn_ref=ref))
        s=stats[split];s['boxes']+=1;s['agreed_code']+=agreed
        s['both_not_qcvn']+=(la==lb==NOT_QCVN);s['both_uncertain']+=(la==lb==UNCERTAIN)
        s['same_label_any']+=(la==lb);s['code_vs_code_disagree']+=(is_code(la) and is_code(lb) and la!=lb)
    total=collections.Counter()
    for s in stats.values():total.update(s)
    stats['ALL']=total
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    MANIFEST.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
    for split in ('TRAIN','VAL','TEST','ALL'):
        s=stats[split];print(split,dict(s),'raw agreement %.4f'%(s['same_label_any']/s['boxes']))
    (OUT/'agreement.json').write_text(json.dumps({k:dict(v) for k,v in stats.items()},indent=1),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('record');r.add_argument('--pass',dest='pass_name',choices=['A','B'],required=True);r.add_argument('--sheet',type=int,required=True)
    r.add_argument('--force',action='store_true');r.add_argument('assignments');r.set_defaults(fn=record)
    sub.add_parser('status').set_defaults(fn=status);sub.add_parser('merge').set_defaults(fn=merge)
    args=ap.parse_args();args.fn(args)


if __name__=='__main__':main()
