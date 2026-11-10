"""Extract a reproducible official TRAIN subset; VAL/TEST are never training inputs."""
from pathlib import Path
import json,random,tarfile,shutil,time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/raw/culane'
OUT=BASE/'train_subset'

def main():
    OUT.mkdir(exist_ok=True)
    lines=(BASE/'test/list/train_gt.txt').read_text().splitlines()
    eligible=[line for line in lines if line.split()[0].startswith('/driver_23_30frame/')]
    random.Random(391).shuffle(eligible)
    selected=eligible[:4096]
    names={line.split()[0].lstrip('/') for line in selected}
    for split in ['val','test']:
        other={line.split()[0].lstrip('/') for line in (BASE/f'test/list/{split}.txt').read_text().splitlines()}
        assert not names & other
    (OUT/'train_gt.txt').write_text('\n'.join(selected)+'\n')
    wanted=names | {str(Path(n).with_suffix('.lines.txt')).replace('\\','/') for n in names}
    masks={line.split()[1].lstrip('/') for line in selected}
    for archive,expected in [('driver_23_30frame.tar.gz',wanted),('laneseg_label_w16.tar.gz',masks)]:
        missing={n for n in expected if not (OUT/n).exists()}
        if not missing:continue
        started=time.time();done=0
        with tarfile.open(BASE/archive,'r|gz') as tar:
            for member in tar:
                name=member.name.removeprefix('./')
                if name not in missing or not member.isfile():continue
                dest=(OUT/name).resolve()
                assert dest.is_relative_to(OUT.resolve())
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
                done+=1
                if done%1000==0:print(archive,done,round(time.time()-started),flush=True)
        remaining=[n for n in expected if not (OUT/n).exists()]
        if remaining:raise RuntimeError(f'{archive}: missing {len(remaining)} files')
    (OUT/'manifest.json').write_text(json.dumps(dict(seed=391,n=len(names),source='official TRAIN driver_23 subset',
        limitation='4096-image subset, not full CULane training; no video labels; VAL/TEST excluded',names=sorted(names)),indent=2))
    print('TRAIN extraction complete',len(names),flush=True)

if __name__=='__main__':main()
