"""Verified CIFAR archives, fixed class-stratified splits and matched sampling."""
import hashlib
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .results import read_json,sha256,write_json

URL='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
ARCHIVE_MD5='eb9058c3a382ffc7106e4002c42a8d85'


def make_splits(fine_labels, validation_per_class, seed):
    rng=np.random.default_rng(seed); train=[]; val=[]
    for fine in np.unique(fine_labels):
        indices=np.flatnonzero(fine_labels==fine); rng.shuffle(indices)
        if not 0 < validation_per_class < len(indices):
            raise ValueError('validation-per-class must leave training data in every class')
        val.extend(indices[:validation_per_class]); train.extend(indices[validation_per_class:])
    return np.sort(train),np.sort(val)


def make_mapping(fine, coarse, fine_names, coarse_names):
    tasks=[]; local=np.full(len(fine_names),-1,dtype=np.int64)
    fine_to_coarse=np.full(len(fine_names),-1,dtype=np.int64)
    for task,name in enumerate(coarse_names):
        ids=sorted(np.unique(fine[coarse==task]).tolist())
        if len(ids)!=5:
            raise ValueError(f'Task {task} must have five fine labels')
        for i,label in enumerate(ids):
            if np.any(coarse[fine==label]!=task):
                raise ValueError('Fine label maps to multiple tasks')
            local[label]=i; fine_to_coarse[label]=task
        tasks.append(dict(id=task,name=name,fine_ids=ids,class_names=[fine_names[i] for i in ids]))
    if np.any(local<0):
        raise ValueError('Mapping does not cover all fine labels')
    return dict(tasks=tasks,fine_to_local=local.tolist(),fine_to_coarse=fine_to_coarse.tolist())


def verify_assets(root):
    root=Path(root); manifest=read_json(root/'manifest.json')
    for filename,expected in manifest['files'].items():
        if sha256(root/filename)!=expected:
            raise ValueError(f'Asset checksum mismatch: {filename}')
    return manifest


def prepare(root, validation_per_class=50, seed=123, archive_path=None):
    root=Path(root)
    if (root/'manifest.json').is_file():
        manifest=verify_assets(root)
        if manifest['split_seed']!=seed or manifest['validation_per_class']!=validation_per_class:
            raise ValueError('Existing split settings differ; choose a new resources directory')
        print(f'Reusing verified CIFAR-100 assets: {root}',flush=True)
        return manifest
    root.mkdir(parents=True,exist_ok=True)
    if any((root/name).exists() for name in ('dataset.npz','split_indices.npz','task_mapping.json')):
        raise FileExistsError('Partial prepared data exists; choose a new resources directory')
    archive=Path(archive_path) if archive_path else root/'cifar-100-python.tar.gz'
    if not archive.exists():
        print('Downloading CIFAR-100 (~169 MB)...',flush=True)
        temporary=archive.with_suffix(archive.suffix+'.part')
        last=[-1]
        def progress(block,size,total):
            percent=min(100,int(block*size*100/max(total,1)))
            if percent//10>last[0]:
                print(f'Download {percent}%',flush=True);last[0]=percent//10
        urllib.request.urlretrieve(URL,temporary,reporthook=progress)
        temporary.replace(archive)
    digest=hashlib.md5()
    with open(archive,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=ARCHIVE_MD5:
        raise ValueError('CIFAR archive failed official MD5 verification; replace the archive and retry')
    # Read only known members; never extract arbitrary archive paths.
    with tarfile.open(archive,'r:gz') as tar:
        def load(name):
            return pickle.load(tar.extractfile('cifar-100-python/'+name),encoding='latin1')
        train,test,meta=load('train'),load('test'),load('meta')
    tr_f=np.asarray(train['fine_labels']); tr_c=np.asarray(train['coarse_labels'])
    te_f=np.asarray(test['fine_labels']); te_c=np.asarray(test['coarse_labels'])
    if len(tr_f)!=50000 or len(te_f)!=10000:
        raise ValueError('Unexpected official CIFAR-100 split sizes')
    mapping=make_mapping(tr_f,tr_c,meta['fine_label_names'],meta['coarse_label_names'])
    if len(mapping['tasks'])!=20 or len(mapping['fine_to_local'])!=100:
        raise ValueError('Expected 20 tasks and 100 fine classes')
    if np.any(np.asarray(mapping['fine_to_coarse'])[te_f]!=te_c):
        raise ValueError('Test task mapping differs from training mapping')
    if not np.all(np.bincount(tr_f,minlength=100)==500) or not np.all(np.bincount(te_f,minlength=100)==100):
        raise ValueError('Unexpected official CIFAR-100 dimensions')
    train_idx,val_idx=make_splits(tr_f,validation_per_class,seed)
    images=np.asarray(train['data'],dtype=np.uint8).reshape(-1,3,32,32)
    test_images=np.asarray(test['data'],dtype=np.uint8).reshape(-1,3,32,32)
    # Training-only channel statistics, accumulated without a huge float copy.
    sums=np.zeros(3); squares=np.zeros(3); count=0
    for start in range(0,len(train_idx),1000):
        block=images[train_idx[start:start+1000]].astype(np.float64)/255
        sums+=block.sum((0,2,3)); squares+=(block*block).sum((0,2,3)); count+=len(block)*32*32
    mean=sums/count; std=np.sqrt(squares/count-mean**2)
    np.savez(root/'dataset.npz',train_images=images,train_fine=tr_f,train_coarse=tr_c,
             test_images=test_images,test_fine=te_f,test_coarse=te_c)
    np.savez(root/'split_indices.npz',train=train_idx,validation=val_idx)
    write_json(root/'task_mapping.json',mapping)
    manifest=dict(dataset='CIFAR-100',source_url=URL,archive_md5=ARCHIVE_MD5,
        archive_sha256=sha256(archive),split_seed=seed,validation_per_class=validation_per_class,
        tasks=20,classes_per_task=5,normalization=dict(mean=mean.tolist(),std=std.tolist()),
        split_counts=dict(train=len(train_idx),validation=len(val_idx),test=len(te_f)),
        files={name:sha256(root/name) for name in ('dataset.npz','split_indices.npz','task_mapping.json')})
    write_json(root/'manifest.json',manifest)
    print('Prepared independent data; no training started.',flush=True)
    return manifest


class Data:
    def __init__(self,root,manifest,include_test=False):
        self.root=Path(root); self.mapping=read_json(self.root/'task_mapping.json')
        self.tasks=len(self.mapping['tasks']);self.classes=len(self.mapping['tasks'][0]['fine_ids'])
        with np.load(self.root/'dataset.npz',allow_pickle=False) as ds, np.load(self.root/'split_indices.npz') as splits:
            images=torch.from_numpy(ds['train_images'])
            coarse=torch.from_numpy(ds['train_coarse'].astype(np.int64))
            local=torch.tensor(self.mapping['fine_to_local'])[torch.from_numpy(ds['train_fine'].astype(np.int64))]
            self.splits={}
            for split in ('train','validation'):
                ids=torch.from_numpy(splits[split].astype(np.int64))
                self.splits[split]=(images,coarse,local,ids)
            if include_test:
                ids=torch.arange(len(ds['test_fine']))
                self.splits['test']=(torch.from_numpy(ds['test_images']),
                    torch.from_numpy(ds['test_coarse'].astype(np.int64)),
                    torch.tensor(self.mapping['fine_to_local'])[torch.from_numpy(ds['test_fine'].astype(np.int64))],ids)
        self.mean=torch.tensor(manifest['normalization']['mean']).reshape(1,3,1,1)
        self.std=torch.tensor(manifest['normalization']['std']).reshape(1,3,1,1)

    def normalize(self,images):
        return (images.float()/255-self.mean)/self.std


class BalancedBatches:
    """Independent sampling/augmentation RNG; projection never changes input draws."""
    def __init__(self,data,batch_size,seed,augment=True):
        self.data,self.batch_size,self.augment=data,batch_size,augment
        _,coarse,_,ids=data.splits['train']
        self.indices=[ids[coarse[ids]==t] for t in range(data.tasks)]
        if any(len(x)==0 for x in self.indices):raise ValueError('Every task needs training images')
        self.sample_rng=torch.Generator().manual_seed(seed+10001)
        self.augment_rng=torch.Generator().manual_seed(seed+20001)
        self.orders=[torch.randperm(len(x),generator=self.sample_rng) for x in self.indices]
        self.positions=[0]*data.tasks

    def next(self):
        chosen=[]
        for t,ids in enumerate(self.indices):
            parts=[];left=self.batch_size
            while left:
                if self.positions[t]==len(ids):
                    self.orders[t]=torch.randperm(len(ids),generator=self.sample_rng);self.positions[t]=0
                take=min(left,len(ids)-self.positions[t]);pos=self.positions[t]
                parts.append(ids[self.orders[t][pos:pos+take]])
                self.positions[t]+=take;left-=take
            chosen.append(torch.cat(parts))
        indices=torch.stack(chosen);images,_,local,_=self.data.splits['train']
        x=images[indices.flatten()];trace=indices.numpy().tobytes()
        if self.augment:
            x=F.pad(x,(4,4,4,4));n=len(x)
            shifts=torch.randint(0,9,(n,2),generator=self.augment_rng)
            flips=torch.rand(n,generator=self.augment_rng)<.5
            # CPU transforms remain identical between methods and across resumes.
            x=torch.stack([im[:,int(s[0]):int(s[0])+32,int(s[1]):int(s[1])+32] for im,s in zip(x,shifts)])
            x[flips]=x[flips].flip(-1)
            trace+=shifts.numpy().tobytes()+flips.numpy().tobytes()
        x=self.data.normalize(x).reshape(self.data.tasks,self.batch_size,3,32,32)
        return x,local[indices],hashlib.sha256(trace).hexdigest()

    def state_dict(self):
        return dict(sample_rng=self.sample_rng.get_state(),augment_rng=self.augment_rng.get_state(),
                    orders=[x.clone() for x in self.orders],positions=list(self.positions))

    def load_state_dict(self,state):
        self.sample_rng.set_state(state['sample_rng']);self.augment_rng.set_state(state['augment_rng'])
        self.orders=[x.clone() for x in state['orders']];self.positions=list(state['positions'])
