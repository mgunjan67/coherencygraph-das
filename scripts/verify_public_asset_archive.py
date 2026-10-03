"""Verify release asset hashes and reject private documents or credential markers."""
from pathlib import Path
import hashlib,json,re,sys,zipfile

def main():
    path=Path(sys.argv[1]);errors=[]
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('RELEASE_MANIFEST.json'))
        assert len(z.namelist())==len(set(z.namelist()))
        for item in manifest['files']:
            name=item['path'];parts=Path(name).parts
            assert not any(p in {'manuscript','production','raw'} for p in parts),name
            assert not name.lower().endswith(('.tex','.docx','.h5','.hdf5','.pem','.key')),name
            data=z.read(name)
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
            if Path(name).suffix in {'.json','.csv','.md','.py','.txt','.xml'}:
                text=data.decode('utf-8',errors='replace')
                if re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN.*PRIVATE KEY-----|sk-proj-[A-Za-z0-9]{20,}',text): errors.append(name)
        assert not errors,errors
    print(json.dumps(dict(passed=True,files=len(manifest['files']),private_document_paths=0,credential_markers=0,scope='Asset checksum and structural/privacy marker scan; not scientific validation'),indent=2))

if __name__=='__main__':main()
