from __future__ import annotations
import copy, json, random, shutil, time, re
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(r'D:\other\DT-RAVLA')
SWAT_PATH=ROOT/'Dataset'/'SWaT'/'merged.csv'
WADI_NORMAL_PATH=ROOT/'Dataset'/'WADI'/'WADI_14days_new.csv'
WADI_ATTACK_PATH=ROOT/'Dataset'/'WADI'/'WADI_attackdataLABLE.csv'
RESULTS=ROOT/'Results'
WADI_ATTACK_HEADER=1
WADI_LABEL_COLUMN='Attack LABLE (1:No Attack, -1:Attack)'
SEEDS=[42,52,62,72,82]
WINDOW,STRIDE,MAX_WINDOWS=64,8,50000
BATCH_SIZE,EPOCHS,PATIENCE=128,25,5
LR,WEIGHT_DECAY=1e-4,1e-4
TOKEN_DIM=18
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def clean_results():
    if RESULTS.exists(): shutil.rmtree(RESULTS)
    for n in ['tables','models','figures','diagnostics']: (RESULTS/n).mkdir(parents=True,exist_ok=True)

def canon(x): return re.sub(r'[^a-z0-9]+','_',str(x).strip().lower()).strip('_')

def read_csv(path,header=0):
    last=None
    for enc in [None,'utf-8','latin1']:
        try:
            kw={'header':header,'low_memory':False}
            if enc: kw['encoding']=enc
            return pd.read_csv(path,**kw)
        except Exception as e: last=e
    raise RuntimeError(f'Could not read {path}: {last}')

def normalize_label(s):
    raw = s.astype(str).str.strip().str.lower()
    num = pd.to_numeric(s, errors='coerce')
    unique_num = set(num.dropna().unique())

    # WADI convention: 1 = normal, -1 = attack.
    # Require a non-empty numeric set. An empty set is otherwise a subset
    # of every set and incorrectly turns text SWaT labels into all-NaN.
    if unique_num and unique_num.issubset({-1, 1}):
        return num.map({1: 0, -1: 1})

    # Standard numeric binary labels.
    if unique_num and unique_num.issubset({0, 1}):
        return num

    # Text labels used by SWaT.
    return raw.map({
        'normal': 0,
        'benign': 0,
        'no attack': 0,
        'attack': 1,
        'anomaly': 1,
        'false': 0,
        'true': 1,
        'n': 0,
        'a': 1,
    })

def infer_features(df, ignored):
    ignored = {canon(x) for x in ignored}
    kept, dropped = [], []

    for c in df.columns:
        if canon(c) in ignored:
            continue

        x = pd.to_numeric(df[c], errors='coerce')
        valid_ratio = float(x.notna().mean())
        unique_count = int(x.nunique(dropna=True))

        # Retain binary/discrete actuators. A feature only needs to be numeric
        # in at least 50% of rows and vary somewhere in the complete file.
        if valid_ratio >= 0.50 and unique_count > 1:
            df[c] = x
            kept.append(c)
        else:
            dropped.append({
                'column': c,
                'valid_ratio': valid_ratio,
                'unique_count': unique_count,
            })

    return kept, dropped

def load_swat():
    df=read_csv(SWAT_PATH); df.columns=[str(c).strip() for c in df.columns]
    cand=[c for c in df.columns if canon(c) in {'normal_attack','label','attack_label','binary_label'}]
    if not cand: raise RuntimeError('SWaT label column not found')
    src=cand[0]; df['binary_label']=normalize_label(df[src]); df=df[df.binary_label.notna()].reset_index(drop=True); df['binary_label']=df.binary_label.astype(int)
    feats, dropped = infer_features(
        df,
        {src, 'binary_label', 'timestamp', 'date', 'time', 'row', 'index'}
    )
    print(
        f"SWaT loaded: rows={len(df)}, features={len(feats)}, "
        f"normal={(df['binary_label']==0).sum()}, attack={(df['binary_label']==1).sum()}"
    )
    if dropped:
        print("SWaT dropped columns:", [d['column'] for d in dropped])
    if not feats:
        raise RuntimeError(
            'No SWaT features. First columns after stripping: '
            + repr(df.columns[:20].tolist())
        )
    return df,feats

def load_wadi():
    normal=read_csv(WADI_NORMAL_PATH,0); attack=read_csv(WADI_ATTACK_PATH,WADI_ATTACK_HEADER)
    normal.columns=[str(c).strip() for c in normal.columns]; attack.columns=[str(c).strip() for c in attack.columns]
    if WADI_LABEL_COLUMN not in attack.columns: raise RuntimeError(f'Missing WADI label {WADI_LABEL_COLUMN!r}')
    attack['binary_label']=normalize_label(attack[WADI_LABEL_COLUMN]); attack=attack[attack.binary_label.notna()].reset_index(drop=True); attack['binary_label']=attack.binary_label.astype(int)
    if attack.binary_label.nunique()!=2: raise RuntimeError('WADI label does not contain two classes')
    nf, normal_dropped = infer_features(
        normal, {'timestamp', 'date', 'time', 'row', 'index'}
    )
    af, attack_dropped = infer_features(
        attack,
        {WADI_LABEL_COLUMN, 'binary_label', 'timestamp', 'date', 'time', 'row', 'index'}
    )
    nm={canon(c):c for c in nf}; am={canon(c):c for c in af}; common=sorted(set(nm)&set(am))
    if len(common)<20: raise RuntimeError(f'Too few aligned WADI features: {len(common)}')
    ndf=pd.DataFrame({k:pd.to_numeric(normal[nm[k]],errors='coerce') for k in common})
    adf=pd.DataFrame({k:pd.to_numeric(attack[am[k]],errors='coerce') for k in common}); adf['binary_label']=attack.binary_label.values
    usable=[]
    for k in common:
        z=pd.concat([ndf[k],adf[k]],ignore_index=True)
        if z.notna().mean()>=0.60 and z.nunique(dropna=True)>1: usable.append(k)
    if len(usable)<20: raise RuntimeError(f'Only {len(usable)} usable WADI features')
    return ndf,adf,usable

def split(df,tr=.6,va=.2):
    n=len(df); a=int(n*tr); b=int(n*(tr+va)); return df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()


def split_class_temporally(df, label_value, tr=.60, va=.20):
    """
    Preserve ordering within a class. This avoids the official merged SWaT
    layout putting all attacks in the final test block while also avoiding
    timestamp-level random shuffling.
    """
    part = df[df['binary_label'] == label_value].copy().reset_index(drop=True)
    n = len(part)
    a = int(n * tr)
    b = int(n * (tr + va))
    return (
        part.iloc[:a].copy(),
        part.iloc[a:b].copy(),
        part.iloc[b:].copy(),
    )


def combine_class_splits(df):
    ntr, nva, nte = split_class_temporally(df, 0)
    atr, ava, ate = split_class_temporally(df, 1)

    # Concatenation is used only to fit preprocessing statistics.
    # Temporal windows are created independently for each class below.
    train_fit = pd.concat([ntr, atr], ignore_index=True)
    return (ntr, atr), (nva, ava), (nte, ate), train_fit


def windows_from_class_parts(prep, normal_df, attack_df, training=False, seed=42):
    normal_x = tokens(prep.transform(normal_df))
    attack_x = tokens(prep.transform(attack_df))

    normal_y = np.zeros(len(normal_df), dtype=np.int64)
    attack_y = np.ones(len(attack_df), dtype=np.int64)

    nw = labelled_windows(normal_x, normal_y)
    aw = labelled_windows(attack_x, attack_y)

    X = np.concatenate([nw[0], aw[0]], axis=0)
    y = np.concatenate([nw[1], aw[1]], axis=0)

    # Shuffle complete windows only for optimization. Rows inside each window
    # remain in their original temporal order.
    if training:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X))
        X, y = X[order], y[order]

    return X, y

class Prep:
    def __init__(self): self.scaler=RobustScaler()
    def fit(self,df,features):
        X=df[features].replace([np.inf,-np.inf],np.nan); med=X.median(numeric_only=True).reindex(features).fillna(0); X=X.fillna(med).fillna(0)
        self.features=(X.nunique()>1)[lambda s:s].index.tolist(); self.medians=med.reindex(self.features).fillna(0)
        if not self.features: raise RuntimeError('No variable features')
        A=np.nan_to_num(X[self.features].to_numpy(),nan=0,posinf=0,neginf=0); self.scaler.fit(A); return self
    def transform(self,df):
        X=df[self.features].replace([np.inf,-np.inf],np.nan).fillna(self.medians).fillna(0)
        A=np.nan_to_num(X.to_numpy(),nan=0,posinf=0,neginf=0); return np.clip(self.scaler.transform(A).astype(np.float32),-10,10)

def tokens(X):
    X=np.clip(np.nan_to_num(X,nan=0,posinf=0,neginf=0),-10,10)
    q=np.quantile(X,[.05,.10,.25,.50,.75,.90,.95],axis=1).T
    out=np.column_stack([X.mean(1),X.std(1),X.min(1),X.max(1),np.median(X,1),np.mean(np.abs(X),1),np.sqrt(np.mean(X**2,1)),np.mean(X>0,1),np.mean(X<0,1),np.mean(X==0,1),q,np.mean(np.abs(np.diff(X,axis=1)),1)])
    return np.nan_to_num(out,nan=0,posinf=0,neginf=0).astype(np.float32)

def labelled_windows(X,y):
    xs,ys=[],[]
    for end in range(WINDOW,len(X)+1,STRIDE):
        xs.append(X[end-WINDOW:end]); ys.append(int(y[end-1]))
        if len(xs)>=MAX_WINDOWS: break
    return np.asarray(xs,np.float32),np.asarray(ys,np.int64)

def normal_windows(X):
    xs=[]
    for end in range(WINDOW,len(X)+1,STRIDE):
        xs.append(X[end-WINDOW:end])
        if len(xs)>=MAX_WINDOWS: break
    return np.asarray(xs,np.float32)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); h=128
        self.inp=nn.Sequential(nn.Linear(TOKEN_DIM,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(.1))
        layer=nn.TransformerEncoderLayer(h,4,h*4,dropout=.1,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,3); self.pool=nn.Sequential(nn.Linear(h,1),nn.Softmax(1))
    def forward(self,x):
        h=self.temporal(self.inp(x)); w=self.pool(h); return (w*h).sum(1),h,w.squeeze(-1)

class SupModel(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder(); self.head=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
    def forward(self,x):
        z,h,w=self.encoder(x); return {'logits':torch.clamp(self.head(z).squeeze(-1),-20,20),'embedding':z,'attention':w}

class OCModel(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder(); self.decoder=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,TOKEN_DIM))
    def forward(self,x):
        z,h,w=self.encoder(x); return {'reconstruction':self.decoder(h),'embedding':z,'attention':w}

def loader(X,y=None,shuffle=False):
    ds=TensorDataset(torch.from_numpy(X)) if y is None else TensorDataset(torch.from_numpy(X),torch.from_numpy(y))
    return DataLoader(ds,batch_size=BATCH_SIZE,shuffle=shuffle)

@torch.no_grad()
def sup_probs(model,X):
    model.eval(); out=[]
    for (xb,) in loader(X): out.append(torch.sigmoid(model(xb.to(DEVICE))['logits']).cpu().numpy())
    return np.concatenate(out)

def train_sup(model,Xtr,ytr,Xv,yv):
    pos=max(1,int(ytr.sum())); neg=max(1,len(ytr)-pos)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=DEVICE)); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    model.to(DEVICE); best=None; bestap=-np.inf; wait=0; start=time.time()
    for _ in range(EPOCHS):
        model.train()
        for xb,yb in loader(Xtr,ytr,True):
            xb,yb=xb.to(DEVICE),yb.float().to(DEVICE); opt.zero_grad(set_to_none=True); loss=lossfn(model(xb)['logits'],yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        p=sup_probs(model,Xv); ap=average_precision_score(yv,p)
        if ap>bestap: bestap=ap; best=copy.deepcopy(model.state_dict()); wait=0
        else:
            wait+=1
            if wait>=PATIENCE: break
    model.load_state_dict(best); return model,time.time()-start

@torch.no_grad()
def rec_scores(model,X):
    model.eval(); out=[]
    for (xb,) in loader(X):
        xb=xb.to(DEVICE); r=model(xb)['reconstruction']; out.append(torch.mean((r-xb)**2,dim=(1,2)).cpu().numpy())
    return np.concatenate(out)

def train_oc(model,Xtr,Xv):
    lossfn=nn.MSELoss(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    model.to(DEVICE); best=None; bestloss=np.inf; wait=0; start=time.time()
    for _ in range(EPOCHS):
        model.train()
        for (xb,) in loader(Xtr,None,True):
            xb=xb.to(DEVICE); opt.zero_grad(set_to_none=True); r=model(xb)['reconstruction']; loss=lossfn(r,xb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        v=rec_scores(model,Xv).mean()
        if v<bestloss: bestloss=v; best=copy.deepcopy(model.state_dict()); wait=0
        else:
            wait+=1
            if wait>=PATIENCE: break
    model.load_state_dict(best); return model,time.time()-start

def threshold(y,s):
    bt=float(np.median(s)); bf=-1
    for t in np.linspace(float(s.min()),float(s.max()),201):
        f=f1_score(y,s>=t,zero_division=0)
        if f>bf: bt,bf=float(t),f
    return bt

def ece(y,p,bins=15):
    edges=np.linspace(0,1,bins+1); v=0.0
    for i in range(bins):
        m=(p>=edges[i])&(p<edges[i+1])
        if m.any(): v+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(v)

def metric_pack(y,s,t,p=None):
    pred=(s>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); eps=1e-12
    if p is None: p=(s-s.min())/(s.max()-s.min()+eps)
    p=np.clip(np.nan_to_num(p,nan=.5),0,1)
    return {'Accuracy':accuracy_score(y,pred),'Balanced_Accuracy':balanced_accuracy_score(y,pred),'Precision':precision_score(y,pred,zero_division=0),'Recall_DR':recall_score(y,pred,zero_division=0),'Specificity':tn/(tn+fp+eps),'F1':f1_score(y,pred,zero_division=0),'F2':fbeta_score(y,pred,beta=2,zero_division=0),'AUROC':roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan,'AUPRC':average_precision_score(y,s) if len(np.unique(y))>1 else np.nan,'MCC':matthews_corrcoef(y,pred),'Cohen_Kappa':cohen_kappa_score(y,pred),'FPR_FAR':fp/(fp+tn+eps),'FNR_MissRate':fn/(fn+tp+eps),'NPV':tn/(tn+fn+eps),'Brier':brier_score_loss(y,p),'ECE':ece(y,p),'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),'Threshold':float(t),'Collapsed_All_Attack':bool(pred.sum()==len(pred)),'Collapsed_All_Normal':bool(pred.sum()==0)}

def mean_std(rows,groups):
    df=pd.DataFrame(rows); nums=[c for c in df.select_dtypes(include=np.number).columns if c!='Seed']; g=df.groupby(groups,dropna=False)
    return df,pd.concat([g[nums].mean().add_suffix('_Mean'),g[nums].std(ddof=1).add_suffix('_Std')],axis=1).reset_index()

def main():
    clean_results(); print('Device:',DEVICE)
    swat,swf=load_swat(); wn,wa,wf=load_wadi()

    # Correct supervised SWaT protocol:
    # split normal and attack samples independently in temporal order;
    # create windows independently; shuffle windows only for training.
    sw_train_parts, sw_val_parts, sw_test_parts, sw_fit_df = combine_class_splits(swat)
    sp = Prep().fit(sw_fit_df, swf)

    swtr = windows_from_class_parts(
        sp, sw_train_parts[0], sw_train_parts[1], training=True, seed=42
    )
    swv = windows_from_class_parts(
        sp, sw_val_parts[0], sw_val_parts[1], training=False
    )
    swte = windows_from_class_parts(
        sp, sw_test_parts[0], sw_test_parts[1], training=False
    )

    if len(np.unique(swtr[1])) < 2:
        raise RuntimeError("SWaT training windows still contain only one class.")
    if len(np.unique(swv[1])) < 2:
        raise RuntimeError("SWaT validation windows still contain only one class.")
    if len(np.unique(swte[1])) < 2:
        raise RuntimeError("SWaT test windows still contain only one class.")

    wntr,wnv,_=split(wn,.70,.15)
    _,wav,wat=split(wa,.20,.20)
    wp=Prep().fit(wntr,wf)
    wtrn=normal_windows(tokens(wp.transform(wntr))); wvn=normal_windows(tokens(wp.transform(wnv)))
    prep_wa=lambda df: labelled_windows(tokens(wp.transform(df)),df.binary_label.to_numpy(np.int64))
    wv,wt=prep_wa(wav),prep_wa(wat)

    diag={
        'WADI_Label_Column':WADI_LABEL_COLUMN,
        'WADI_Header':WADI_ATTACK_HEADER,
        'WADI_Aligned_Features':len(wf),
        'SWaT_Usable_Features':len(swf),
        'SWaT_Window_Counts':{
            'Train':{'Normal':int((swtr[1]==0).sum()),'Attack':int((swtr[1]==1).sum())},
            'Validation':{'Normal':int((swv[1]==0).sum()),'Attack':int((swv[1]==1).sum())},
            'Test':{'Normal':int((swte[1]==0).sum()),'Attack':int((swte[1]==1).sum())},
        },
        'SWaT_Window_Attack_Ratios':{
            'Train':float(swtr[1].mean()),
            'Validation':float(swv[1].mean()),
            'Test':float(swte[1].mean())
        },
        'WADI_Window_Attack_Ratios':{
            'Validation':float(wv[1].mean()),
            'Test':float(wt[1].mean())
        }
    }
    (RESULTS/'diagnostics'/'data_protocol.json').write_text(json.dumps(diag,indent=2),encoding='utf-8'); print(json.dumps(diag,indent=2))

    bench,abl,trans,rob=[],[],[],[]
    A=np.concatenate([swtr[0].mean(1),swtr[0].std(1),swtr[0].min(1),swtr[0].max(1)],1)
    V=np.concatenate([swv[0].mean(1),swv[0].std(1),swv[0].min(1),swv[0].max(1)],1)
    B=np.concatenate([swte[0].mean(1),swte[0].std(1),swte[0].min(1),swte[0].max(1)],1)
    for name,m in {'LogisticRegression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,solver='saga',class_weight='balanced',random_state=42)),'RandomForest':RandomForestClassifier(n_estimators=300,max_depth=20,class_weight='balanced',random_state=42,n_jobs=-1)}.items():
        m.fit(A,swtr[1])
        pv=m.predict_proba(V)[:,1]
        t=threshold(swv[1],pv)
        p=m.predict_proba(B)[:,1]
        r=metric_pack(swte[1],p,t,p)
        r.update({'Dataset':'SWaT','Model':name,'Seed':-1})
        bench.append(r)
        joblib.dump(m,RESULTS/'models'/f'SWaT_{name}.joblib')

    for seed in SEEDS:
        print('Seed',seed); seed_all(seed)
        sm=SupModel(); sm,sec=train_sup(sm,swtr[0],swtr[1],swv[0],swv[1]); pv=sup_probs(sm,swv[0]); st=threshold(swv[1],pv); pt=sup_probs(sm,swte[0]); full=metric_pack(swte[1],pt,st,pt); full.update({'Dataset':'SWaT','Model':'DT-RAVLA','Seed':seed,'Train_Time_sec':sec,'Parameters':sum(p.numel() for p in sm.parameters())}); bench.append(full); a=copy.deepcopy(full); a['Configuration']='Full'; abl.append(a)

        seed_all(seed); nm=SupModel(); nm.encoder.temporal=nn.Identity(); nm,nsec=train_sup(nm,swtr[0],swtr[1],swv[0],swv[1]); npv=sup_probs(nm,swv[0]); nt=threshold(swv[1],npv); npt=sup_probs(nm,swte[0]); ar=metric_pack(swte[1],npt,nt,npt); ar.update({'Dataset':'SWaT','Configuration':'No_Transformer','Seed':seed,'Train_Time_sec':nsec,'Parameters':sum(p.numel() for p in nm.parameters())}); abl.append(ar)

        seed_all(seed); om=OCModel(); om,osec=train_oc(om,wtrn,wvn); sv=rec_scores(om,wv[0]); ot=threshold(wv[1],sv); ss=rec_scores(om,wt[0]); wr=metric_pack(wt[1],ss,ot); wr.update({'Dataset':'WADI','Model':'DT-RAVLA-OneClass','Seed':seed,'Train_Time_sec':osec,'Parameters':sum(p.numel() for p in om.parameters())}); bench.append(wr)

        zp=sup_probs(sm,wt[0]); zr=metric_pack(wt[1],zp,st,zp); zr.update({'Direction':'SWaT_to_WADI','Protocol':'Zero-shot','Target_Label_Fraction':0.0,'Seed':seed}); trans.append(zr)

        for cond,sev in [('Clean',0),('Gaussian_5pct',.05),('Gaussian_10pct',.10),('Missing_10pct',.10),('Missing_20pct',.20)]:
            Xc=wt[0].copy(); rng=np.random.RandomState(seed)
            if 'Gaussian' in cond: Xc+=rng.normal(0,sev,Xc.shape).astype(np.float32)
            elif 'Missing' in cond: Xc[rng.rand(*Xc.shape)<sev]=0
            cs=rec_scores(om,Xc); rr=metric_pack(wt[1],cs,ot); rr.update({'Dataset':'WADI','Condition':cond,'Seed':seed}); rob.append(rr)

        torch.save(sm.state_dict(),RESULTS/'models'/f'SWaT_DT_RAVLA_seed{seed}.pt'); torch.save(om.state_dict(),RESULTS/'models'/f'WADI_DT_RAVLA_seed{seed}.pt')

    for rows,groups,rawname,sumname in [(bench,['Dataset','Model'],'Table_2_Raw_All_Seeds.csv','Table_2_Benchmark_Mean_Std.csv'),(abl,['Dataset','Configuration'],'Table_3a_Raw_Ablation.csv','Table_3a_Ablation_Mean_Std.csv'),(trans,['Direction','Protocol','Target_Label_Fraction'],'Table_3b_Raw_Cross_Dataset.csv','Table_3b_Cross_Dataset_Mean_Std.csv'),(rob,['Dataset','Condition'],'Table_3c_Raw_Robustness.csv','Table_3c_Robustness_Mean_Std.csv')]:
        raw,summ=mean_std(rows,groups); raw.to_csv(RESULTS/'tables'/rawname,index=False); summ.to_csv(RESULTS/'tables'/sumname,index=False)
    print('Completed:',RESULTS)

if __name__=='__main__': main()
