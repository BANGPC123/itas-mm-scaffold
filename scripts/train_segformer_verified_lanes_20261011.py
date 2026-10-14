"""Train SegFormer-B0 supervised candidate on CULane or human-verified Vietnam.

By default trains on CULane native masks (not on AI pseudo labels). Vietnam
training mode requires verified_manifest.jsonl made ONLY by human-label gate.
Never overwrite selected UFLDv2; metrics refer ONLY to dataset-specific VAL.
"""
from pathlib import Path
import argparse,json,random,time,datetime,cv2,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]

class DenseLanes(Dataset):
    def __init__(self,items,mode):
        self.items=items;self.mode=mode
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        img_path,label=self.items[i]
        img=cv2.imread(str(img_path))
        if img is None:raise ValueError('Unreadable source '+str(img_path))
        h,w=img.shape[:2]
        if self.mode=='culane':
            mask=cv2.imread(str(label),cv2.IMREAD_UNCHANGED)
            if mask is None:raise ValueError('Missing GT mask: '+str(label))
            mask=(mask>0).astype('uint8')
        else:
            mask=cv2.imread(str(label),cv2.IMREAD_UNCHANGED)
            if mask is None:raise ValueError('Missing approved GT mask')
            if np.any(~np.isin(mask,[0,1,2,255])):raise ValueError('Illegal class mask')
        img=cv2.resize(img,(512,288),interpolation=cv2.INTER_AREA)
        mask=cv2.resize(mask,(512,288),interpolation=cv2.INTER_NEAREST)
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return torch.from_numpy(rgb.transpose(2,0,1).copy()),torch.from_numpy(mask.astype('int64'))

def load_culane_train():
    base=ROOT/'data/raw/culane/train_subset'
    imgs=sorted(base.glob('driver*/*/*.jpg'))
    # training subset masks built from real CULane lane GT
    pairs=[]
    for p in imgs:
        mask=base/'laneseg_label_w16'/p.relative_to(base).with_suffix('.png')
        if mask.exists():pairs.append((p,mask))
    random.Random(391).shuffle(pairs)
    return pairs[:640]

def load_culane_val():
    base=ROOT/'data/raw/culane/val'
    names=[s.split()[0].lstrip('/') for s in
        (ROOT/'data/raw/culane/test/list/val.txt').read_text().splitlines() if s.strip()][:100]
    cache=ROOT/'output/retraining_20261011/segformer_culane_b0/official_val_gt_masks'
    cache.mkdir(parents=True,exist_ok=True)
    pairs=[]
    for i,name in enumerate(names):
        path=base/name
        if not path.exists():continue
        maskpath=cache/(str(i).zfill(4)+'.png')
        if not maskpath.exists():
            img=cv2.imread(str(path))
            if img is None:continue
            h,w=img.shape[:2]
            mask=np.zeros((h,w),np.uint8)
            lines=(base/Path(name).with_suffix('.lines.txt')).read_text().splitlines()
            for line in lines:
                numbers=np.fromstring(line,sep=' ')
                if len(numbers)>=4:
                    pts=numbers.reshape(-1,2).round().astype(np.int32)
                    cv2.polylines(mask,[pts],False,1,16)
            cv2.imwrite(str(maskpath),mask)
        pairs.append((path,maskpath))
    return pairs

def load_human(split):
    p=ROOT/'data/processed/lane_human_verified_20261011/verified_manifest.jsonl'
    if not p.is_file():raise RuntimeError('Human-verified GT not imported: run scripts/import_human_lane_masks_20261011.py')
    rows=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
    actual=[r for r in rows if r['split']==split and r.get('human_confirmed') is True]
    if len(actual)<(50 if split=='TRAIN' else 15):raise RuntimeError('Too few human-confirmed '+split+' labels')
    return [(Path(r['image']),Path(r['mask'])) for r in actual]

def metrics(model,loader,device):
    model.eval();tp=fp=fn=0;miou_inter=np.zeros(3,np.float64);miou_union=np.zeros(3,np.float64)
    with torch.inference_mode():
        for images,gt in loader:
            images=images.to(device);gt=gt.to(device)
            with torch.autocast(device_type='cuda',enabled=device.type=='cuda'):
                logits=model(pixel_values=images).logits
            logits=F.interpolate(logits.float(),size=gt.shape[-2:],mode='bilinear',align_corners=False)
            pred=logits.argmax(1)
            valid=gt!=255
            for c in (1,2):
                matches=(pred==c)&valid
                truth=(gt==c)&valid
                miou_inter[c]+=int((matches&truth).sum())
                miou_union[c]+=int((matches|truth).sum())
            tp+=int(((pred==1)&(gt==1)&valid).sum())
            fp+=int(((pred==1)&(gt!=1)&valid).sum())
            fn+=int(((pred!=1)&(gt==1)&valid).sum())
    return {'paint_pixel_precision':tp/max(1,tp+fp),
        'paint_pixel_recall':tp/max(1,tp+fn),
        'paint_pixel_f1':2*tp/max(1,2*tp+fp+fn),
        'paint_pixel_iou':float(miou_inter[1]/max(1,miou_union[1])),
        'road_edge_iou':float(miou_inter[2]/max(1,miou_union[2])) if miou_union[2] else None,
        'tp':tp,'fp':fp,'fn':fn,'gt_road_edge_pixels':int(miou_union[2])}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',choices=('culane','vietnam'),default='culane')
    ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--batch',type=int,default=6)
    ap.add_argument('--lr',type=float,default=6e-5)
    args=ap.parse_args()
    torch.manual_seed(391);np.random.seed(391);random.seed(391);torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.data=='culane':
        train=load_culane_train();val=load_culane_val()
        source='nvidia/segformer-b0-finetuned-ade-512-512'
        save=ROOT/'output/retraining_20261011/segformer_culane_b0'
    else:
        train=load_human('TRAIN');val=load_human('VAL_INTERNAL')
        source=str(ROOT/'output/retraining_20261011/segformer_culane_b0/best')
        if not (Path(source)/'config.json').is_file():
            raise RuntimeError('No frozen CULane source candidate to fine-tune')
        save=ROOT/'output/retraining_20261011/segformer_vietnam_human_b0'
    if len(train)<50 or len(val)<15:raise RuntimeError('Insufficient supervised labels')
    if (save/'training_report.json').is_file():raise FileExistsError('Do not overwrite saved candidate')
    save.mkdir(parents=True,exist_ok=True)
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.data=='culane':
        model=SegformerForSemanticSegmentation.from_pretrained(source,num_labels=3,ignore_mismatched_sizes=True)
    else:
        model=SegformerForSemanticSegmentation.from_pretrained(source)
    model.to(dev)
    dl=DataLoader(DenseLanes(train,args.data),batch_size=args.batch,shuffle=True,num_workers=0)
    vl=DataLoader(DenseLanes(val,args.data),batch_size=args.batch,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.015)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    class_w=torch.tensor([.08,1.,.8],device=dev)
    started=time.perf_counter();history=[];best=-1.
    for epoch in range(args.epochs):
        model.train();losses=[]
        for step,(images,gt) in enumerate(dl):
            images=images.to(dev);gt=gt.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=dev.type=='cuda'):
                output=model(pixel_values=images).logits
                output=F.interpolate(output.float(),size=gt.shape[-2:],mode='bilinear',align_corners=False)
                loss=F.cross_entropy(output,gt,weight=class_w,ignore_index=255)
                target=(gt==1).float()
                valid=(gt!=255).float()
                p=output.softmax(1)[:,1]*valid
                dice=(2*(p*target).sum()+1)/((p+target).sum()+1)
                loss=loss+.35*(1-dice)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach().cpu()))
        score=metrics(model,vl,dev)
        row={'epoch':epoch+1,'mean_train_loss':round(float(np.mean(losses)),5),
             'validation':score,'training_samples':len(train),'val_samples':len(val)}
        history.append(row)
        print('SEGFORMER_EPOCH',json.dumps(row,ensure_ascii=False),flush=True)
        if score['paint_pixel_f1']>best:
            best=score['paint_pixel_f1']
            model.save_pretrained(str(save/'best'),safe_serialization=True)
        (save/'training_history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'status':'EXPERIMENTAL_SEGFORMER_CULANE_GT' if args.data=='culane' else 'EXPERIMENTAL_SEGFORMER_HUMAN_VIETNAM_GT',
        'training_dataset':args.data,'train_samples':len(train),'val_samples':len(val),
        'epochs':args.epochs,'batch':args.batch,
        'best_paint_pixel_f1':best,'pixel_F1_is_NOT_lane_level_CULane_metric':True,
        'trained_on_AI_pseudolabels':False,'provenance':source,
        'vietnam_field_test_accuracy':None if args.data=='culane' else 'Requires different independent source',
        'road_edge_GT_available':args.data=='vietnam',
        'selected_production_model_changed':False,
        'training_minutes':round((time.perf_counter()-started)/60,2)}
    (save/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SEGFORMER_TRAIN_COMPLETE',json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
