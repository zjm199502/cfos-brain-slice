"""Reassemble the exact validated ZIPs and update the release candidate via a normal fast-forward."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import hashlib, io, json, os, subprocess

data=Path(__file__).resolve().parent
spec=json.loads((data/'release.json').read_bytes())
target='release/fosfolio-0.9.19'
base=spec['base']
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*args, payload=None):
    return subprocess.check_output(['git',*args], input=payload).decode().strip()
def blob(payload):
    return git('hash-object','-w','--stdin',payload=payload)
entries={}
for name, asset in spec['assets'].items():
    raw=b''.join((data/p).read_bytes() for p in asset['parts'])
    assert sha(raw)==asset['sha256'] and len(raw)==asset['size'], name
    with ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('patch.json'))
        assert name==f"cfos-patch-{manifest['version']}.zip"
        assert set(z.namelist())=={'patch.json', *manifest['files']}
        assert all(sha(z.read(p))==h for p,h in manifest['files'].items())
    entries[name]=blob(raw)
    print(name,asset['sha256'],flush=True)
git('fetch','--depth=1','--filter=blob:none','origin','refs/heads/'+target)
assert git('rev-parse','FETCH_HEAD')==base, 'Release candidate moved'
with TemporaryDirectory() as temp:
    os.environ['GIT_INDEX_FILE']=str(Path(temp)/'index')
    git('-c','core.sparseCheckout=false','read-tree',base)
    paths={p:entries[n] for p,n in spec['files'].items()}
    for f in (data/'files').rglob('*'):
        if f.is_file():
            path=f.relative_to(data/'files').as_posix()
            if path.endswith('latest.json'):
                m=json.loads(f.read_bytes())
                archive=spec['assets'][m['patch']['file']]
                assert m['patch']['size']==archive['size'] and m['patch']['sha256']==archive['sha256']
                assert spec['files'][str(Path(path).parent / m['patch']['file'])]==m['patch']['file']
            paths[path]=blob(f.read_bytes())
    for p,oid in paths.items():
        git('update-index','--add','--cacheinfo','100644',oid,p)
    tree=git('write-tree')
commit=git('-c','user.name=github-actions[bot]','-c','user.email=41898282+github-actions[bot]@users.noreply.github.com',
           'commit-tree',tree,'-p',base,'-m','发布 FosFolio 0.9.19 双端图谱配准更新与旧版升级提醒')
assert set(git('-c','core.quotePath=false','diff','--name-only',base,commit).splitlines())==set(paths)
git('push','origin',commit+':refs/heads/'+target)
print('Release candidate',commit)
