import sys, bisect, subprocess
sys.path.insert(0,'.')
import nxloader
mods,_,_=nxloader.load_program(sys.argv[1])
tab=[]
for m in mods:
    for v,n in m.syms.items(): tab.append((m.base+v,n,m.name))
tab.sort(); keys=[t[0] for t in tab]
def dem(n):
    try: return subprocess.run(['c++filt',n],capture_output=True,text=True).stdout.strip()
    except Exception: return n
for a in sys.argv[2:]:
    x=int(a,16); i=bisect.bisect_right(keys,x)-1
    if i>=0: print('%s -> %s!%s+0x%x'%(a,tab[i][2],dem(tab[i][1]),x-tab[i][0]))
