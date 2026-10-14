"""Download two licensed, different-source Vietnam road videos from Wikimedia Commons.

Retrieves immutable original URLs and license metadata via Commons API,
streams raw bytes with SHA256 and no re-encoding. No legal/field GT claims.
"""
from pathlib import Path
import hashlib,json,time,requests,datetime,html,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/external/vietnam_commons_street_20261011'
OUT.mkdir(parents=True,exist_ok=True)
SOURCES=[
 ('hanoi_2026_CC0','File:Hanoi2.webm'),
 ('saigon_govap_2019_CCBYSA','File:Saigon traffic-oVbn3HeLDA0.webm'),
]
session=requests.Session()
session.headers.update({'User-Agent':'ITASMM-ResearchDatasetCurator/1.0 (OpenAI user research; license attribution kept)'})
rows=[]
for key,title in SOURCES:
    params={'action':'query','titles':title,'prop':'imageinfo',
            'iiprop':'url|size|sha1|extmetadata|timestamp','format':'json'}
    resp=session.get('https://commons.wikimedia.org/w/api.php',params=params,timeout=30)
    resp.raise_for_status()
    page=next(iter(resp.json()['query']['pages'].values()))
    if 'imageinfo' not in page:raise RuntimeError('Missing API image info: '+title)
    info=page['imageinfo'][0]
    meta=info.get('extmetadata',{})
    license_id=html.unescape(re.sub('<[^>]+>','',meta.get('LicenseShortName',{}).get('value','')))
    author=html.unescape(re.sub('<[^>]+>','',meta.get('Artist',{}).get('value','')))
    credit=html.unescape(re.sub('<[^>]+>','',meta.get('Credit',{}).get('value','')))
    print('MEDIA',key,'size',info.get('size'),'license',license_id,'author',author[:90],flush=True)
    if key.startswith('hanoi') and 'CC0' not in license_id.upper():raise ValueError('Expected CC0 license')
    if key.startswith('saigon') and 'BY-SA' not in license_id.upper():raise ValueError('Expected CC BY-SA license')
    final=OUT/(key+'.webm')
    if final.exists():raise FileExistsError('Existing original video; refuse overwrite '+str(final))
    partial=OUT/(key+'.partial')
    digest=hashlib.sha256();length=0
    with session.get(info['url'],stream=True,timeout=(30,180)) as r:
        r.raise_for_status()
        with partial.open('wb') as dest:
            for chunk in r.iter_content(chunk_size=1024*512):
                if not chunk:continue
                digest.update(chunk);length+=len(chunk)
                dest.write(chunk)
    if length!=int(info['size']):
        raise RuntimeError('Downloaded size differs from upstream metadata: '+str(length)+' != '+str(info['size']))
    partial.rename(final)
    row={'id':key,'commons_page':'https://commons.wikimedia.org/wiki/'+title.replace(' ','_'),
        'original_title':title,'original_url':info['url'],
        'author':author,'credit':credit,'license_id':license_id,
        'license_url':meta.get('LicenseUrl',{}).get('value',''),
        'commons_sha1':info['sha1'],'size_bytes':length,'sha256':digest.hexdigest(),
        'filename':final.name,'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'author_attribution_required': 'BY' in license_id.upper(),
        'status':'UNREVIEWED_SOURCE_VIDEO, NOT ANNOTATED, NOT FIELD TEST'}
    rows.append(row)
    print('SOURCE_SAVED',json.dumps({'id':key,'bytes':length,'sha256':row['sha256']},ensure_ascii=False),flush=True)
(OUT/'SOURCES.json').write_text(json.dumps({'sources':rows,
 'purpose':'Independent-source candidate videos for AI annotation and human verification; not a finished independent test.',
 'license_notes':'Follow Commons source attribution/share-alike terms; source metadata must accompany any derivative dataset.'},
 indent=2,ensure_ascii=False),encoding='utf-8')
print('COMPLETE',len(rows),flush=True)
