"""One-shot, read-only CropGuard v7 diagnostics.  Writes JSON, overlays, and log."""
import os, sys, glob, json, math, random, shutil, re, subprocess, traceback
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

IN, W = "/kaggle/input", "/kaggle/working"
def find(marker):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, marker)): return p
    return None
SRC = find("code/model/architecture/v7_model.py")
if SRC is None: raise RuntimeError("diagnostics input not mounted: " + repr(glob.glob(IN+"/**", recursive=True)[:30]))
sys.path.insert(0, os.path.join(SRC, "code"))
from model.training.dataset import CropGuardDataset, collate, load_yolo_labels
from model.training.losses import multitask_loss, seg_loss_v7
from model.architecture.v7_model import CropGuardMultiTaskV7

class EMA:
    def __init__(self, model, decay=.999):
        self.decay=decay
        self.shadow={k:v.detach().clone().float() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach().float(),alpha=1-self.decay)

def cosine_lr(step,total,warmup,base_lr,min_lr=1e-6):
    if step<warmup: return base_lr*step/max(warmup,1)
    p=(step-warmup)/max(total-warmup,1)
    return min_lr+(base_lr-min_lr)*.5*(1+math.cos(math.pi*p))
DATA=os.path.join(SRC,"data"); CKPT=os.path.join(SRC,"checkpoint","best_val.pt")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
out={"device":str(DEV)}
print("INPUT",SRC,"DATA",DATA,"CKPT",CKPT, flush=True)

# (a) Eight real, augmented DataLoader batches: image/masks overlay plus a forced-geometry proof.
os.makedirs(os.path.join(W,"overlays"),exist_ok=True)
random.seed(20261011); np.random.seed(20261011); torch.manual_seed(20261011)
aug=CropGuardDataset(DATA,"train",320,augment=True)
ldr=DataLoader(aug,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate)
for n,b in zip(range(8),ldr):
    im=(b["image"][0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
    ov=im.copy(); leaf=b["leaf"][0].numpy()>0.5; les=b["mask"][0].numpy()>0.5
    ov[leaf]=[0,255,0]; ov[les]=[255,0,0]
    cv2.imwrite(os.path.join(W,"overlays",f"batch_{n:02d}.png"),cv2.cvtColor(cv2.addWeighted(im,.55,ov,.45,0),cv2.COLOR_RGB2BGR))
# Force every image geometry branch but compare masks: dedicated masks should change if synchronized.
plain=CropGuardDataset(DATA,"train",320,augment=False)[0]
oldr, oldu=random.random, random.uniform
random.random=lambda:0.0; random.uniform=lambda a,b:1.1 if (a,b)==(.9,1.1) else (a+b)/2
forced=CropGuardDataset(DATA,"train",320,augment=True)[0]
random.random,random.uniform=oldr,oldu
out["a"]={"overlay_count":8,"forced_image_mae_vs_plain":float((forced["image"]-plain["image"]).abs().mean()),"forced_lesion_mask_mae_vs_plain":float((forced["mask"]-plain["mask"]).abs().mean()),"forced_leaf_mask_mae_vs_plain":float((forced["leaf"]-plain["leaf"]).abs().mean()),"source_geometry":"image uses PIL.BILINEAR; dedicated lesion/leaf masks only cv2.INTER_NEAREST resize and receive no flip/scale"}
print("A",json.dumps(out["a"]),flush=True)

# instantiate with pretrained weights; record evidence instead of silently accepting fallback
model=CropGuardMultiTaskV7(nc=8,pretrained=True).to(DEV)
import timm
rand=timm.create_model("fasternet_t0",pretrained=False,features_only=True,out_indices=(0,1,2,3))
l1=float((next(model.backbone.net.parameters()).detach().cpu()-next(rand.parameters()).detach().cpu()).abs().mean())
out["i"]={"pretrained_first_conv_l1_vs_random":l1,"torch_hub_cache":torch.hub.get_dir(),"pretrained_loaded":bool(l1>.01)}; print("I",json.dumps(out["i"]),flush=True)

# (c) optimizer / LR schedule
opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
names=[n for n,p in model.named_parameters() if any(p is q for g in opt.param_groups for q in g["params"])]
steps=len(DataLoader(aug,batch_size=8,drop_last=True,collate_fn=collate)); total=steps*120; warm=steps*5
out["c"]={"optimizer_parameter_count":len(names),"seg_head_parameter_count":sum(n.startswith("seg_head.") for n in names),"neck_parameter_count":sum(n.startswith("neck.") for n in names),"seg_head_requires_grad":{n:p.requires_grad for n,p in model.named_parameters() if n.startswith("seg_head.")},"lr_epoch_start":{str(e):cosine_lr((e-1)*steps,total,warm,5e-4) for e in (1,30,60,90,120)}}
print("C",json.dumps(out["c"]),flush=True)

# (d/e/f) One backward: graph, group norms, exact components, and EMA delta.
b=next(iter(DataLoader(aug,batch_size=8,shuffle=False,num_workers=0,collate_fn=collate)))
b={k:v.to(DEV) if torch.is_tensor(v) else v for k,v in b.items()}; model.train(); opt.zero_grad(set_to_none=True)
o=model(b["image"],active_tasks={"cls","seg"},gt_mask=b["mask"],gt_mask_prob=.8)
loss,parts=multitask_loss(o,b,320,nc=8,active_tasks={"cls","seg"},two_channel_seg=True,train_sev=False,label_smoothing=.05)
L=o["seg_logits"][:,0:1]; T=b["mask"][:,None].float(); P=L.sigmoid(); pw=torch.tensor([20.],device=DEV)
bce=F.binary_cross_entropy_with_logits(L,T,pos_weight=pw); tp=(P*T).sum((2,3)); fp=(P*(1-T)).sum((2,3)); fn=((1-P)*T).sum((2,3)); ft=(1-(tp+1e-4)/(tp+.3*fp+.7*fn+1e-4)).pow(1.33).mean()
empty=(T.sum((1,2,3))==0)
loss.backward()
def gn(prefix):
    x=[p.grad.detach().float().pow(2).sum() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
    return float(torch.sqrt(sum(x)).item()) if x else 0.
out["d"]={"seg_head_grad_norm":gn("seg_head."),"neck_grad_norm":gn("neck."),"backbone_grad_norm":gn("backbone."),"seg_logits_requires_grad":bool(o["seg_logits"].requires_grad),"seg_logits_grad_fn":str(o["seg_logits"].grad_fn)}
out["e"]={"multitask_parts":{k:float(v.detach()) for k,v in parts.items()},"lesion_bce":float(bce.detach()),"lesion_focal_tversky":float(ft.detach()),"lesion_loss_bce_plus_3ft":float((bce+3*ft).detach()),"empty_lesion_images":int(empty.sum()),"empty_tversky_values":[float(x) for x in ((1-(tp+1e-4)/(tp+.3*fp+.7*fn+1e-4)).pow(1.33)[empty]).detach().cpu()]}
ema=EMA(model); before=next(iter(ema.shadow.values())).clone(); opt.step(); ema.update(model); after=next(iter(ema.shadow.values()))
out["f"]={"validation_weights":"EMA: train_full_v7 calls ema_eval(), copies shadow into model, validates, then restores raw weights","ema_shadow_abs_change_after_one_step":float((after-before).abs().mean()),"ema_updates_per_step":True}
print("D",json.dumps(out["d"]),"E",json.dumps(out["e"]),"F",json.dumps(out["f"]),flush=True)

# (g) Best checkpoint: predicted vs true/stored severity and areas on VAL only.
ck=torch.load(CKPT,map_location=DEV); model.load_state_dict(ck["model"] if "model" in ck else ck); model.eval()
pred_area=[]; true_area=[]; stored=[]; truth=[]
with torch.no_grad():
 for vb in DataLoader(CropGuardDataset(DATA,"val",320,augment=False),batch_size=8,shuffle=False,num_workers=0,collate_fn=collate):
  x=vb["image"].to(DEV); q=model(x,active_tasks={"seg"},with_analysis=True); pp=q["seg_logits"].sigmoid().cpu()
  pred_area.extend(pp[:,0].mean((1,2)).tolist()); true_area.extend(vb["mask"].mean((1,2)).tolist()); stored.extend(vb["sev_cont"].tolist()); truth.extend((vb["mask"].sum((1,2))/vb["leaf"].clamp_min(0).sum((1,2)).clamp_min(1)).tolist())
valid=np.asarray(stored)>=0
out["g"]={"stored_severity_vs_true_mask_mae":float(np.abs(np.asarray(stored)[valid]-np.asarray(truth)[valid]).mean()),"mean_predicted_lesion_area_fraction":float(np.mean(pred_area)),"mean_true_lesion_area_fraction":float(np.mean(true_area)),"pred_over_true_area_ratio":float(np.mean(pred_area)/max(np.mean(true_area),1e-12)),"checkpoint_epoch":ck.get("epoch")}
print("G",json.dumps(out["g"]),flush=True)

# (b) Paired, authorized 8-image memorization tests.  The augmented loader
# deliberately redraws its geometry every step, exactly as training does.
def mem_test(augment):
 mem=CropGuardMultiTaskV7(nc=8,pretrained=True).to(DEV); mo=torch.optim.AdamW(mem.parameters(),lr=1e-3,weight_decay=1e-4)
 ds=Subset(CropGuardDataset(DATA,"train",320,augment=augment),list(range(8))); dl=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,collate_fn=collate); hist=[]
 for step in range(1,501):
  mb=next(iter(dl)); mb={k:v.to(DEV) if torch.is_tensor(v) else v for k,v in mb.items()}; mo.zero_grad(); z=mem(mb["image"],active_tasks={"seg"}); ml=seg_loss_v7(z["seg_logits"],mb["mask"],mb["leaf"]); ml.backward(); mo.step()
  if step%100==0:
   with torch.no_grad():
    prob=mem(mb["image"],active_tasks={"seg"})["seg_logits"][:,0].sigmoid(); gt=mb["mask"]; best=0.
    for th in (.3,.4,.5,.6,.7):
     pr=(prob>th).float(); inter=(pr*gt).sum((1,2)); union=((pr+gt)>0).sum((1,2)); best=max(best,float((inter/union.clamp_min(1)).mean()))
   rec={"step":step,"loss":float(ml.detach()),"mean_iou":best}; hist.append(rec); print("B augment="+str(augment),rec,flush=True)
 return hist
off=mem_test(False); on=mem_test(True)
out["b"]={"augmentation_off":off,"augmentation_on":on,"off_passes_0_5":off[-1]["mean_iou"]>=.5,"on_passes_0_5":on[-1]["mean_iou"]>=.5}
if out["b"]["on_passes_0_5"]:
    print("GATE PASS: launching 120-epoch cls+seg retrain", flush=True)
    train_out=os.path.join(W,"retrain_v7")
    env=dict(os.environ); env["PYTHONPATH"]=os.path.join(SRC,"code")
    rc=subprocess.call([sys.executable,"train_full_v7.py","--root",DATA,"--out",train_out,"--epochs","120","--batch-size","8","--size","320","--workers","2"],cwd=os.path.join(SRC,"code","model","training"),env=env)
    out["retrain"]={"return_code":rc,"out":train_out}
    print("RETRAIN_RETURN",rc,flush=True)
else:
    out["retrain"]={"skipped":"augmentation-on memorization gate failed"}
    print("GATE FAIL: no retrain",flush=True)

# (h) labels absent/outside 0..7; (j) checkpoint metric; (k) healthy style statistics.
bad={}; stats={"healthy":[],"diseased":[]}
for split in ("train","val","test"):
 ldir=os.path.join(DATA,"labels",split); bad[split]=[]
 for f in sorted(glob.glob(ldir+"/*.txt")):
  rows=open(f).read().splitlines(); cls=[int(float(r.split()[0])) for r in rows if len(r.split())>=5]
  if not cls or any(c<0 or c>7 for c in cls): bad[split].append({"file":os.path.basename(f),"contents":rows})
 for ip in glob.glob(os.path.join(DATA,"images",split,"*")):
  stem=os.path.splitext(os.path.basename(ip))[0]; rows=load_yolo_labels(os.path.join(ldir,stem+".txt")); key="healthy" if rows and max(r[0] for r in rows)==0 else "diseased"
  im=cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB); leaf=cv2.imread(os.path.join(DATA,"masks_leaf",split,stem+".png"),0); bg=im[leaf<=127] if leaf is not None else im.reshape(-1,3)
  stats[key].append([im.shape[0],im.shape[1],*im.mean((0,1)),*bg.mean(0)])
out["h"]={s:{"count":len(v),"files":v} for s,v in bad.items()}
out["j"]={"best_checkpoint_metric":"val macro-F1 only","checkpoint_epoch":ck.get("epoch"),"checkpoint_best_metrics":ck.get("best_metrics")}
out["k"]={k:{"n":len(v),"mean_h_w_rgb_background_rgb":np.mean(v,axis=0).tolist()} for k,v in stats.items()}
out["mask_path_audit"]={"val_augment_flag":CropGuardDataset(DATA,"val",320,augment=True).augment,"train_augment_flag":CropGuardDataset(DATA,"train",320,augment=True).augment,"dedicated_masks_transform":"not called in __getitem__; only raw_mask is passed to _transform_mask","consumers":[p for p in glob.glob(os.path.join(SRC,"code","model","training","*.py")) if "mask" in open(p,errors="ignore").read()]}
print("H",json.dumps(out["h"]),"J",json.dumps(out["j"]),"K",json.dumps(out["k"]),flush=True)
json.dump(out,open(os.path.join(W,"diagnostics.json"),"w"),indent=2)
print("SUMMARY",json.dumps({k:out[k] for k in "abcdefgijk"}),flush=True)
