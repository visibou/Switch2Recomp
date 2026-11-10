"""Names the ntdll and ucrtbase time of the busiest threads in a host profile (Windows exports).

  python tools/host_prof_modules.py <host_profile.txt>
"""
import re,struct,bisect,collections,sys
def exports(p):
    d=open(p,'rb').read()
    pe=struct.unpack_from('<I',d,0x3c)[0]; nsec=struct.unpack_from('<H',d,pe+6)[0]; optsz=struct.unpack_from('<H',d,pe+20)[0]
    opt=pe+24; ex=struct.unpack_from('<I',d,opt+112)[0]
    secs=[struct.unpack_from('<8sIIII',d,opt+optsz+40*i) for i in range(nsec)]
    def off(r):
        for n,vs,va,rs,rp in secs:
            if va<=r<va+max(vs,rs): return r-va+rp
    e=off(ex); nf,nn,af,an,ao=struct.unpack_from('<5I',d,e+20)
    out=[]
    for i in range(nn):
        nr=struct.unpack_from('<I',d,off(an)+4*i)[0]; o=struct.unpack_from('<H',d,off(ao)+2*i)[0]
        rva=struct.unpack_from('<I',d,off(af)+4*o)[0]
        nm=d[off(nr):d.index(b'\0',off(nr))].decode()
        out.append((rva,nm))
    return sorted(out)
mods={'ntdll.dll':exports(r'C:\Windows\System32\ntdll.dll'),'ucrtbase.dll':exports(r'C:\Windows\System32\ucrtbase.dll')}
txt=open(sys.argv[1],errors='replace').read()
for b in re.split(r'(?m)^(?=thread )',txt)[1:]:
    h=b.split('\n',1)[0]
    m=re.search(r"(\d+)% of a core, (\d+) samples",h)
    if not m or int(m.group(1))<int(sys.argv[2] if len(sys.argv)>2 else 70): continue
    tot=int(m.group(2)); print(h[:50])
    for mod,ex in mods.items():
        keys=[e[0] for e in ex]; c=collections.Counter()
        for n,off in re.findall(r'(?m)^\s+(\d+) '+re.escape(mod)+r'\+([0-9a-f]+)',b):
            i=bisect.bisect_right(keys,int(off,16))-1; c[ex[i][1]]+=int(n)
        print('  ',mod,[(k,round(100*v/tot,1)) for k,v in c.most_common(9)])
