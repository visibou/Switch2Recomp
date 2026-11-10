/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* fs: RomFS (built on the fly from an extracted romfs/ folder, or a romfs.bin), plus save data and SD card mapped
   to normal folders */
#include "rt.h"
#include <ctype.h>

#define FS_NOTFOUND RC(2,1)
#define FS_EXISTS   RC(2,2)

/* ============================================================ RomFS virtual */
typedef struct { char* path; u64 off, size; int logged; } RFile;

static size_t g_romdir_len;
static RFile* rfiles; static int nrfiles;
static u8* rmeta; static u64 rmeta_size;        /* header + tables sit at the start of the image */
static u64 rdata_off, rtotal;
static FILE* rraw;                               /* plain romfs.bin */

char g_romfs_dir_override[1024];

#if !defined(_WIN32)
#include <unistd.h>
#endif
static PMutex g_romlock;

typedef struct DNode { char* name; int parent; int first_child, next_sib, first_file; u32 off; } DNode;

typedef struct FNode { char* name; int parent, next_sib; u32 off; u64 size; char* host; } FNode;

static DNode* dn; static int ndn, capdn; static FNode* fn; static int nfn, capfn;

static int add_dir(const char* name, int parent){ if(ndn==capdn){ capdn=capdn?capdn*2:256; dn=realloc(dn,sizeof(DNode)*capdn); }
  DNode* d=&dn[ndn]; d->name=strdup(name); d->parent=parent; d->first_child=-1; d->next_sib=-1; d->first_file=-1; return ndn++; }

static int add_file(const char* name, int parent, u64 size, const char* host){ if(nfn==capfn){ capfn=capfn?capfn*2:1024; fn=realloc(fn,sizeof(FNode)*capfn); }
  FNode* f=&fn[nfn]; f->name=strdup(name); f->parent=parent; f->next_sib=-1; f->size=size; f->host=strdup(host); return nfn++; }

static int cmpname(const void* a, const void* b){ return strcmp(*(char* const*)a,*(char* const*)b); }

static void scan(const char* host, int di){
  PDir d; if(plat_dir_open(&d,host)) return;
  char** names=NULL; int* isdir=NULL; u64* sizes=NULL; int n=0, cap=0;
  while(plat_dir_next(&d)){ if(n==cap){ cap=cap?cap*2:64; names=realloc(names,sizeof(char*)*cap); isdir=realloc(isdir,sizeof(int)*cap); sizes=realloc(sizes,8*cap); }
    names[n]=strdup(d.name); isdir[n]=d.is_dir; sizes[n]=d.size; n++; }
  plat_dir_close(&d);
  /* dirs first, then alphabetical, so the layout is stable */
  for(int pass=0;pass<2;pass++){
    int lastd=-1, lastf=-1;
    char** idx=malloc(sizeof(char*)*(n?n:1)); int m=0;
    for(int i=0;i<n;i++) if(isdir[i]==(pass==0)) idx[m++]=names[i];
    qsort(idx,m,sizeof(char*),cmpname);
    for(int k=0;k<m;k++){ int i=0; while(names[i]!=idx[k]) i++;
      char p[2048]; snprintf(p,sizeof p,"%s/%s",host,names[i]);
      if(pass==0){ int c=add_dir(names[i],di); if(lastd<0) dn[di].first_child=c; else dn[lastd].next_sib=c; lastd=c; scan(p,c); }
      else { int f=add_file(names[i],di,sizes[i],p); if(lastf<0) dn[di].first_file=f; else fn[lastf].next_sib=f; lastf=f; } }
    free(idx);
  }
  for(int i=0;i<n;i++) free(names[i]); free(names); free(isdir); free(sizes);
}

static u32 rhash(u32 parent, const char* s){ u32 h=parent^123456789u; for(;*s;s++){ h=(h>>5)|(h<<27); h^=(u8)*s; } return h; }

static u32 hcount(u32 n){ if(n<3) return 3; if(n<19) return n|1; u32 c=n; while(!(c%2)||!(c%3)||!(c%5)||!(c%7)||!(c%11)||!(c%13)||!(c%17)) c++; return c; }

static int rf_cmp(const void* a, const void* b){ const RFile* x=a,*y=b; return x->off<y->off? -1 : x->off>y->off; }

static void romfs_build(const char* dir){
  add_dir("",0); dn[0].parent=0; g_romdir_len=strlen(dir);
  scan(dir,0);
  /* table offsets */
  u32 doff=0; for(int i=0;i<ndn;i++){ dn[i].off=doff; doff+=0x18+((strlen(dn[i].name)+3)&~3u); }
  u32 foff=0; for(int i=0;i<nfn;i++){ fn[i].off=foff; foff+=0x20+((strlen(fn[i].name)+3)&~3u); }
  u32 dhc=hcount(ndn), fhc=hcount(nfn);
  u64 hdr=0x50, dht=hdr, dmt=dht+dhc*4, fht=dmt+doff, fmt=fht+fhc*4, end=fmt+foff;
  rdata_off=(end+0x1FF)&~0x1FFull;
  rmeta_size=rdata_off; rmeta=calloc(1,rmeta_size);
  u64 H[10]={0x50,dht,dhc*4ull,dmt,doff,fht,fhc*4ull,fmt,foff,rdata_off}; memcpy(rmeta,H,0x50);
  u32* dh=(u32*)(rmeta+dht); for(u32 i=0;i<dhc;i++) dh[i]=0xFFFFFFFFu;
  u32* fh=(u32*)(rmeta+fht); for(u32 i=0;i<fhc;i++) fh[i]=0xFFFFFFFFu;
  u64 data=0;
  rfiles=calloc(nfn?nfn:1,sizeof(RFile));
  for(int i=0;i<nfn;i++){ fn[i].size=fn[i].size; data=(data+0xF)&~0xFull;
    /* an empty file has the same offset as the next one, if we list it romfs_read finds it instead and reads the
       real file back as padding (zeros). it never gets read anyway */
    if(fn[i].size){ rfiles[nrfiles].path=fn[i].host; rfiles[nrfiles].off=rdata_off+data; rfiles[nrfiles].size=fn[i].size; nrfiles++; }
    u8* e=rmeta+fmt+fn[i].off; u32 w[2]={dn[fn[i].parent].off, fn[i].next_sib>=0? fn[fn[i].next_sib].off : 0xFFFFFFFFu};
    memcpy(e,w,8); memcpy(e+8,&data,8); memcpy(e+16,&fn[i].size,8);
    u32 h=rhash(dn[fn[i].parent].off,fn[i].name)%fhc; u32 nl=(u32)strlen(fn[i].name);
    memcpy(e+0x18,&fh[h],4); fh[h]=fn[i].off; memcpy(e+0x1C,&nl,4); memcpy(e+0x20,fn[i].name,nl);
    data+=fn[i].size; }
  for(int i=0;i<ndn;i++){ u8* e=rmeta+dmt+dn[i].off;
    u32 w[4]={dn[dn[i].parent].off, dn[i].next_sib>=0? dn[dn[i].next_sib].off : 0xFFFFFFFFu, dn[i].first_child>=0? dn[dn[i].first_child].off : 0xFFFFFFFFu,
      dn[i].first_file>=0? fn[dn[i].first_file].off : 0xFFFFFFFFu};
    memcpy(e,w,16); u32 h=rhash(i? dn[dn[i].parent].off : 0,dn[i].name)%dhc; u32 nl=(u32)strlen(dn[i].name);
    memcpy(e+16,&dh[h],4); dh[h]=dn[i].off; memcpy(e+20,&nl,4); memcpy(e+24,dn[i].name,nl); }
  rtotal=rdata_off+data;
  qsort(rfiles,nrfiles,sizeof(RFile),rf_cmp);
  LOG("RomFS: %d folders, %d files, %.1f MB\n",ndn,nfn,(double)rtotal/1048576.0);
}

void romfs_init(void){
  pm_init(&g_romlock);
  char p[1200];
  if(g_romfs_dir_override[0] && plat_is_dir(g_romfs_dir_override)){ LOG("RomFS: %s\n",g_romfs_dir_override); romfs_build(g_romfs_dir_override); return; }
  snprintf(p,sizeof p,"%sromfs.bin",g_datadir); rraw=fopen(p,"rb");
  if(rraw){ rtotal=plat_file_size(p); LOG("RomFS: image %s (%.1f MB)\n",p,(double)rtotal/1048576.0); return; }
  snprintf(p,sizeof p,"%sromfs",g_datadir);
  if(plat_is_dir(p)){ LOG("RomFS: folder %s\n",p); romfs_build(p); return; }
  LOG("RomFS: none (folder %s)\n",p);
  char q[1200]; snprintf(q,sizeof q,"%sexefs/main.nro",g_datadir);
  if(!plat_file_size(q)){
    char m[1600]; snprintf(m,sizeof m,"The game's romfs is missing (the data: models, textures, sound).\n\n"
      "Extract it from your dump and copy the folder to:\n%s\n\nThe game will close when it tries to read its data.",p);
#if defined(_WIN32)
    extern void win_fatal_box(const char*); win_fatal_box(m);
#else
    plat_message(m,1,1);
#endif
    LOG("%s\n",m); }
}

u64 romfs_size(void){ return rtotal; }

/* cache of open host files */
#define NOPEN 48
static struct { const char* path; FILE* f; u64 last; } openf[NOPEN]; static u64 openclk;

static FILE* rf_open(const char* path){
  int lru=0;
  for(int i=0;i<NOPEN;i++){ if(openf[i].path==path){ openf[i].last=++openclk; return openf[i].f; } if(openf[i].last<openf[lru].last) lru=i; }
  if(openf[lru].f) fclose(openf[lru].f);
  openf[lru].f=fopen(path,"rb"); openf[lru].path=path; openf[lru].last=++openclk; return openf[lru].f; }

int romfs_read(u64 off, void* dst, u64 size){
  u8* d=dst;
  if(rraw){ pm_lock(&g_romlock); int r=plat_pread(rraw,d,size,off); pm_unlock(&g_romlock); return r<0? -1 : 0; }
  if(off<rmeta_size){ u64 n=rmeta_size-off; if(n>size) n=size; memcpy(d,rmeta+off,n); d+=n; off+=n; size-=n; }
  pm_lock(&g_romlock);
  while(size){
    int lo=0, hi=nrfiles-1, k=-1;
    while(lo<=hi){ int m=(lo+hi)/2; if(rfiles[m].off<=off){ k=m; lo=m+1; } else hi=m-1; }
    if(k<0){ memset(d,0,size); break; }
    RFile* f=&rfiles[k];
    if(off>=f->off+f->size){ /* inter-file padding */
      u64 nx= k+1<nrfiles? rfiles[k+1].off : rtotal; u64 n=nx>off? nx-off : size; if(n>size) n=size; if(!n) n=size; memset(d,0,n); d+=n; off+=n; size-=n; continue; }
    u64 n=f->off+f->size-off; if(n>size) n=size;
#ifdef SWR_HAVE_VP9
    { extern void vp9_note_file(const char* path); vp9_note_file(f->path); }
#endif
    if(!f->logged){ f->logged=1; LOG("romfs: %s (%.1f KB)\n",f->path+g_romdir_len,(double)f->size/1024.0);
      extern void input_rec_file(const char*); input_rec_file(f->path+g_romdir_len); }
    FILE* fp=rf_open(f->path);
    if(!fp || plat_pread(fp,d,n,off-f->off)<(int)n){ memset(d,0,n); }
    d+=n; off+=n; size-=n;
  }
  pm_unlock(&g_romlock);
  return 0;
}

/* ============================================================ IStorage (RomFS) */
static int romstorage(SvcObj* o, IpcCtx* c){
  switch(c->cmd){
    case 0: { u64 off=in64(c,0), sz=in64(c,8), a, bs; buf_out(c,0,&a,&bs); if(sz>bs) sz=bs;
      if(off>rtotal) return RC(2,6001); if(off+sz>rtotal) sz=rtotal-off;
      romfs_read(off,GP(a),sz); return 0; }
    case 4: out64(c,rtotal); return 0;
    case 1: case 2: case 3: return RC(2,6300);
    case 5: { u8 z[0x40]; memset(z,0,sizeof z); outbytes(c,z,0x40); return 0; }
  }
  return stub_ok(o,c); }

/* ============================================================ PC folders as IFileSystem */
typedef struct { char root[1024]; } FsData;

typedef struct { FILE* f; char path[1100]; u32 mode; } FileData;

typedef struct { PDir d; int ok; u32 filter; char path[1100]; int done; } DirData;

static void read_path(IpcCtx* c, int idx, char* out, u32 max){
  u64 a,s; if(!buf_xs(c,idx,&a,&s) || !a){ buf_in(c,idx,&a,&s); }
  u32 n=0; if(a){ const char* p=(const char*)GP(a); while(n<max-1 && n<0x300 && p[n]){ out[n]=p[n]; n++; } } out[n]=0; }

static void host_path(FsData* fs, const char* p, char* out, size_t max){
  while(*p=='/') p++;
  snprintf(out,max,"%s/%s",fs->root,p);
  size_t l=strlen(out); while(l>0 && (out[l-1]=='/')) out[--l]=0;
  for(char* q=out;*q;q++) if(*q=='\\') *q='/';
}

static void mkdirs(const char* path){ char t[1100]; strncpy(t,path,1099); t[1099]=0; for(char* p=t+1;*p;p++){ if(*p=='/'){ *p=0; plat_mkdir(t); *p='/'; } } plat_mkdir(t); }

/* horizon rejects paths that climb out of the mount and so do we, if not a game bug like
   DeleteDirectoryRecursively("/../..") would delete files outside the save folder */
static int path_escapes(const char* p){
  for(const char* q=p;*q;){ while(*q=='/'||*q=='\\') q++; const char* e=q; while(*e && *e!='/' && *e!='\\') e++;
    if((e-q==2 && q[0]=='.' && q[1]=='.') || memchr(q,':',(size_t)(e-q))) return 1;
    q=e; }
  return 0; }

static void copy_tree(const char* src, const char* dst){
  PDir d; if(plat_dir_open(&d,src)) return; mkdirs(dst);
  while(plat_dir_next(&d)){ char s[1600], t[1600]; snprintf(s,sizeof s,"%s/%s",src,d.name); snprintf(t,sizeof t,"%s/%s",dst,d.name);
    if(d.is_dir){ copy_tree(s,t); continue; }
    FILE* fi=fopen(s,"rb"); if(!fi) continue; FILE* fo=fopen(t,"wb"); if(!fo){ fclose(fi); continue; }
    static char buf[1<<16]; size_t n; while((n=fread(buf,1,sizeof buf,fi))>0) fwrite(buf,1,n,fo);
    fclose(fi); fclose(fo); }
  plat_dir_close(&d); }
void fs_copy_tree(const char* src, const char* dst){ copy_tree(src,dst); }   /* input_rec.c: the save at the start of a recording */

static int file_disp(SvcObj* o, IpcCtx* c){
  FileData* f=o->data;
  switch(c->cmd){
    case 0: { u64 off=in64(c,8), sz=in64(c,16), a, bs; buf_out(c,0,&a,&bs); if(sz>bs) sz=bs;
      int r=plat_pread(f->f,GP(a),sz,off); u64 n= r<0? 0 : (u64)r; out64(c,n); return 0; }   /* positioned read (write traps: plat_pread) */
    case 1: { u64 off=in64(c,8), sz=in64(c,16), a, bs; buf_in(c,0,&a,&bs); if(sz>bs) sz=bs;
      fseek(f->f,(long)off,SEEK_SET); fwrite(GP(a),1,sz,f->f); if(in32(c,0)&1) fflush(f->f); return 0; }
    case 2: fflush(f->f); return 0;
    case 3: { u64 n=in64(c,0); fflush(f->f); fseek(f->f,0,SEEK_END); u64 cur=(u64)ftell(f->f);
      if(n>cur){ u64 k=n-cur; static const u8 z[4096]; while(k){ u64 m=k>4096?4096:k; fwrite(z,1,m,f->f); k-=m; } fflush(f->f); }
      else if(n<cur){ /* truncate = rewrite */ u8* buf=malloc(n?n:1); fseek(f->f,0,SEEK_SET); if(fread(buf,1,n,f->f)){} fclose(f->f); f->f=fopen(f->path,"wb+"); fwrite(buf,1,n,f->f); fflush(f->f); free(buf); }
      return 0; }
    case 4: { fflush(f->f); fseek(f->f,0,SEEK_END); out64(c,(u64)ftell(f->f)); return 0; }
    case 5: { u8 z[0x40]; memset(z,0,sizeof z); outbytes(c,z,0x40); return 0; }
  }
  return stub_ok(o,c); }

static void file_destroy(SvcObj* o){ FileData* f=o->data; if(f->f) fclose(f->f); }

static int dir_disp(SvcObj* o, IpcCtx* c){
  DirData* d=o->data;
  switch(c->cmd){
    case 0: { u64 a,s; buf_out(c,0,&a,&s); u64 n=0, max=s/0x310;
      while(n<max && d->ok && !d->done){
        if(!plat_dir_next(&d->d)){ d->done=1; break; }
        if(d->d.is_dir && !(d->filter&1)) continue; if(!d->d.is_dir && !(d->filter&2)) continue;
        u8* e=GP(a+n*0x310); memset(e,0,0x310); strncpy((char*)e,d->d.name,0x300); e[0x304]=d->d.is_dir? 0 : 1; u64 sz=d->d.is_dir? 0 : d->d.size; memcpy(e+0x308,&sz,8); n++; }
      out64(c,n); return 0; }
    case 1: { PDir t; u64 n=0; if(!plat_dir_open(&t,d->path)){ while(plat_dir_next(&t)){ if(t.is_dir? (d->filter&1) : (d->filter&2)) n++; } plat_dir_close(&t); } out64(c,n); return 0; }
  }
  return stub_ok(o,c); }

static void dir_destroy(SvcObj* o){ DirData* d=o->data; if(d->ok) plat_dir_close(&d->d); }

/* save data lookups that fail: a game that waits on a file we don't have shows up here */
static void fs_miss(const char* what, const char* p){ static int n; if(n<sw2_cap(40)){ n++; LOG("fs: %s %s -> not found\n",what,p); } }

static int fs_disp(SvcObj* o, IpcCtx* c){
  FsData* fs=o->data; char p[0x301], hp[1100], p2[0x301], hp2[1100];
  if(c->cmd<=15 && c->cmd!=10){ read_path(c,0,p,sizeof p); if(path_escapes(p)) return FS_NOTFOUND; }
  if(c->cmd==5 || c->cmd==6){ read_path(c,1,p2,sizeof p2); if(path_escapes(p2)) return FS_NOTFOUND; }
  switch(c->cmd){
    case 0: { read_path(c,0,p,sizeof p); host_path(fs,p,hp,sizeof hp); u64 sz=in64(c,8);
      FILE* t=fopen(hp,"rb"); if(t){ fclose(t); return FS_EXISTS; }
      t=fopen(hp,"wb"); if(!t) return FS_NOTFOUND; if(sz){ fseek(t,(long)sz-1,SEEK_SET); fputc(0,t); } fclose(t); return 0; }
    case 1: read_path(c,0,p,sizeof p); host_path(fs,p,hp,sizeof hp); return remove(hp)? FS_NOTFOUND : 0;
    case 2: read_path(c,0,p,sizeof p); host_path(fs,p,hp,sizeof hp); if(plat_is_dir(hp)) return FS_EXISTS; return plat_mkdir(hp)? FS_NOTFOUND : 0;
    case 3: case 4: case 13: { read_path(c,0,p,sizeof p); host_path(fs,p,hp,sizeof hp);
      /* simple recursive delete */
      extern void rm_rf(const char*, int keep); rm_rf(hp,c->cmd==13); return 0; }
    case 5: case 6: read_path(c,0,p,sizeof p); read_path(c,1,p2,sizeof p2); host_path(fs,p,hp,sizeof hp); host_path(fs,p2,hp2,sizeof hp2); return rename(hp,hp2)? FS_NOTFOUND : 0;
    case 7: { read_path(c,0,p,sizeof p); host_path(fs,p,hp,sizeof hp);
      if(plat_is_dir(hp) || !strcmp(p,"/") || !p[0]){ out32(c,0); return 0; }
      FILE* t=fopen(hp,"rb"); if(!t){ fs_miss("GetEntryType",p); return FS_NOTFOUND; } fclose(t); out32(c,1); return 0; }
    case 8: { read_path(c,0,p,sizeof p); host_path(fs,p,hp,sizeof hp); u32 mode=in32(c,0);
      FILE* t=fopen(hp,(mode&2)? "rb+" : "rb"); if(!t){ fs_miss("OpenFile",p); return FS_NOTFOUND; }
      SvcObj* f=svcobj_new("IFile",file_disp,sizeof(FileData)); FileData* fd=f->data; fd->f=t; fd->mode=mode; strcpy(fd->path,hp); f->destroy=file_destroy;
      out_obj(c,f); return 0; }
    case 9: { read_path(c,0,p,sizeof p); host_path(fs,p,hp,sizeof hp);
      if(!plat_is_dir(hp)){ fs_miss("OpenDirectory",p); return FS_NOTFOUND; }
      SvcObj* d=svcobj_new("IDirectory",dir_disp,sizeof(DirData)); DirData* dd=d->data; dd->filter=in32(c,0); if(!dd->filter) dd->filter=3; strcpy(dd->path,hp);
      dd->ok=!plat_dir_open(&dd->d,hp); d->destroy=dir_destroy; out_obj(c,d); return 0; }
    case 10: return 0;
    case 11: out64(c,4ull<<30); return 0;
    case 12: out64(c,8ull<<30); return 0;
    case 14: { u8 z[0x20]; memset(z,0,sizeof z); outbytes(c,z,0x20); return 0; }
    case 15: return 0;
    case 16: out64(c,0); return 0;
  }
  return stub_ok(o,c); }

SvcObj* hostfs_new(const char* root){
  SvcObj* o=svcobj_new("IFileSystem",fs_disp,sizeof(FsData)); FsData* d=o->data; strncpy(d->root,root,1023);
  size_t l=strlen(d->root); while(l && (d->root[l-1]=='/'||d->root[l-1]=='\\')) d->root[--l]=0;
  mkdirs(d->root); return o; }

#if defined(_WIN32)
void rm_rf(const char* p, int keep){ PDir d; if(!plat_dir_open(&d,p)){ while(plat_dir_next(&d)){ char q[1600]; snprintf(q,sizeof q,"%s/%s",p,d.name); if(d.is_dir) rm_rf(q,0); else remove(q); } plat_dir_close(&d); } if(!keep) RemoveDirectoryA(p); }
#else
#include <unistd.h>

void rm_rf(const char* p, int keep){ PDir d; if(!plat_dir_open(&d,p)){ while(plat_dir_next(&d)){ char q[1600]; snprintf(q,sizeof q,"%s/%s",p,d.name); if(d.is_dir) rm_rf(q,0); else remove(q); } plat_dir_close(&d); } if(!keep) rmdir(p); }
#endif

static int savereader(SvcObj* o, IpcCtx* c){ (void)o; if(c->cmd==0){ out64(c,0); return 0; } return 0; }

/* system archives (NgWord, fonts...) aren't available, hand out an empty but valid RomFS */
static u8 g_emptyrom[0x200];

static void emptyrom_init(void){ static int done; if(done) return; done=1; u8* b=g_emptyrom;
  u64 H[10]={0x50, 0x50,4, 0x54,0x18, 0x6C,4, 0x70,0, 0x200}; memcpy(b,H,0x50);
  u32 dh=0; memcpy(b+0x50,&dh,4);
  u32 root[6]={0,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0}; memcpy(b+0x54,root,0x18);
  u32 fh=0xFFFFFFFFu; memcpy(b+0x6C,&fh,4); }

static int emptystorage(SvcObj* o, IpcCtx* c){ (void)o;
  switch(c->cmd){
    case 0: { u64 off=in64(c,0), sz=in64(c,8), a, bs; buf_out(c,0,&a,&bs); if(sz>bs) sz=bs; u8* d=GP(a);
      for(u64 i=0;i<sz;i++) d[i]= off+i<sizeof g_emptyrom? g_emptyrom[off+i] : 0; return 0; }
    case 4: out64(c,sizeof g_emptyrom); return 0;
    case 5: { u8 z[0x40]; memset(z,0,sizeof z); outbytes(c,z,0x40); return 0; }
  }
  return RC(2,6300); }

/* system archives the game reads at boot get a small fake RomFS, like yuzu's system_archive/. NgWord (0806) and
   NgWord2 (0823) are behind nn::ngc::ProfanityFilter, it aborts when version.dat is missing (mario strikers:
   battle league stopped in GetContentVersion at boot because of this) */
typedef struct { const char* name; const u8* data; u32 size; } SynFile;

static u32 rfs_hash(u32 parent, const char* n){ u32 h=parent^123456789u; for(;*n;n++){ h=(h>>5)|(h<<27); h^=(u8)*n; } return h; }

static u8* synrom_build(const SynFile* f, int n, u64* out_size){
  u32 fmeta_sz=0; for(int i=0;i<n;i++) fmeta_sz+=0x20+((u32)(strlen(f[i].name)+3)&~3u);
  u32 nb= n? (u32)n : 1;
  u64 dh=0x50, dm=dh+4, fh=dm+0x18, fm=fh+4ull*nb, data=(fm+fmeta_sz+15)&~15ull, dsz=0;
  for(int i=0;i<n;i++) dsz+=(f[i].size+15)&~15u;
  u64 tot=data+dsz; u8* b=calloc(1,(size_t)tot); if(!b) return NULL;
  u64 H[10]={0x50,dh,4,dm,0x18,fh,4ull*nb,fm,fmeta_sz,data}; memcpy(b,H,0x50);
  u32 z=0; memcpy(b+dh,&z,4);
  u32 root[6]={0,0xFFFFFFFFu,0xFFFFFFFFu, n? 0u : 0xFFFFFFFFu,0xFFFFFFFFu,0}; memcpy(b+dm,root,0x18);
  for(u32 i=0;i<nb;i++){ u32 e=0xFFFFFFFFu; memcpy(b+fh+4*i,&e,4); }
  u32 off=0; u64 doff=0;
  for(int i=0;i<n;i++){ u32 nl=(u32)strlen(f[i].name), es=0x20+((nl+3)&~3u); u8* e=b+fm+off;
    u32 sib= i+1<n? off+es : 0xFFFFFFFFu; u64 sz=f[i].size;
    u32 bk=rfs_hash(0,f[i].name)%nb, prev; memcpy(&prev,b+fh+4*bk,4);
    memcpy(e+0,&z,4); memcpy(e+4,&sib,4); memcpy(e+8,&doff,8); memcpy(e+16,&sz,8); memcpy(e+24,&prev,4); memcpy(e+28,&nl,4); memcpy(e+32,f[i].name,nl);
    memcpy(b+fh+4*bk,&off,4);
    memcpy(b+data+doff,f[i].data,f[i].size); doff+=(f[i].size+15)&~15u; off+=es; }
  *out_size=tot; return b; }

static int synstorage(SvcObj* o, IpcCtx* c){
  u8* img=(u8*)o->data; u64 tot=o->u[0];
  switch(c->cmd){
    case 0: { u64 off=in64(c,0), sz=in64(c,8), a, bs; buf_out(c,0,&a,&bs); if(sz>bs) sz=bs; u8* d=GP(a);
      for(u64 i=0;i<sz;i++) d[i]= off+i<tot? img[off+i] : 0; return 0; }
    case 4: out64(c,tot); return 0;
    case 5: { u8 zz[0x40]; memset(zz,0,sizeof zz); outbytes(c,zz,0x40); return 0; }
  }
  return RC(2,6300); }

static const u8 k_ng_word_txt[]={0xFE,0xFF,0x00,0x5E,0x00,0x76,0x00,0x65,0x00,0x72,0x00,0x79,0x00,0x62,0x00,0x61,0x00,0x64,0x00,0x77,0x00,0x6F,0x00,0x72,0x00,0x64,0x00,0x24,0x00,0x0A}; /* "^verybadword$" */

static const u8 k_ng1_version[]={0,0,0,0x20}, k_ng2_version[]={0,0,0,0x1A};

static const u8 k_ng2_ac[]={0x1F,0x8B,0x08,0x08,0x00,0x00,0x00,0x00,0x02,0xFF,0x61,0x63,0x72,0x61,0x77,0x00,0x63,0x60,0x18,0xD9,0x00,0x00,0x58,0x85,0x96,0x0D,0x00,0x01,0x00,0x00};

/* NULL when there is no synthetic version of that archive */
static SvcObj* sysdata_open(u64 id){
  static char names[64][24]; SynFile f[64]; int n=0, k=0;
  if(id==0x0100000000000806ull){
    for(int i=0;i<16;i++){ snprintf(names[k],24,"%d.txt",i); f[n++]=(SynFile){names[k++],k_ng_word_txt,sizeof k_ng_word_txt}; }
    f[n++]=(SynFile){"common.txt",k_ng_word_txt,sizeof k_ng_word_txt};
    f[n++]=(SynFile){"version.dat",k_ng1_version,4}; }
  else if(id==0x0100000000000823ull){
    for(int i=0;i<16;i++){ const char* sfx[3]={"b1","b2","not_b"};
      for(int j=0;j<3;j++){ snprintf(names[k],24,"ac_%d_%s_nx",i,sfx[j]); f[n++]=(SynFile){names[k++],k_ng2_ac,sizeof k_ng2_ac}; } }
    f[n++]=(SynFile){"ac_common_b1_nx",k_ng2_ac,sizeof k_ng2_ac}; f[n++]=(SynFile){"ac_common_b2_nx",k_ng2_ac,sizeof k_ng2_ac};
    f[n++]=(SynFile){"ac_common_not_b_nx",k_ng2_ac,sizeof k_ng2_ac}; f[n++]=(SynFile){"ac_b1_nx",k_ng2_ac,sizeof k_ng2_ac};
    f[n++]=(SynFile){"ac_b2_nx",k_ng2_ac,sizeof k_ng2_ac}; f[n++]=(SynFile){"ac_not_b_nx",k_ng2_ac,sizeof k_ng2_ac};
    f[n++]=(SynFile){"version.dat",k_ng2_version,4}; }
  else return NULL;
  u64 tot=0; u8* img=synrom_build(f,n,&tot); if(!img) return NULL;
  SvcObj* s=svcobj_new("IStorage(system)",synstorage,0); s->data=img; s->u[0]=tot; return s; }

static int fsp_disp(SvcObj* o, IpcCtx* c){
  char p[1200];
  switch(c->cmd){
    case 1: return 0;
    case 18: snprintf(p,sizeof p,"%ssdmc",g_datadir); out_obj(c,hostfs_new(p)); return 0;
    case 51: case 52: case 53: {
      /* in: u8 space, pad[7], then attr = { u64 program_id, uid[16], u64 system_save_id, u8 type, u8 rank, u16 index } */
      u8 type=in8(c,8+0x20);
      /* System 0, Account 1, Bcat 2, Device 3, Temporary 4, Cache 5, SystemBcat 6 */
      const char* kind= type==1? "user" : (type==2||type==6)? "bcat" : type==3? "device" : type==5? "cache" : type==0? "system" : "temp";
      snprintf(p,sizeof p,"%ssave/%s",g_datadir,kind);
      /* SWR_SAVE_DIR: another save folder (replays of an input recording start from a copy of the save it was made with
         and must not touch the player's own) */
      { const char* sd=getenv("SWR_SAVE_DIR"); if(sd && *sd){ snprintf(p,sizeof p,"%s/%s",sd,kind); mkdirs(p); } }
      /* each save type gets its own folder. games recompiled with older runtimes kept every save in save/system,
         so the first time an account or device save opens I start it from a copy of that folder */
      if((type==1 || type==3) && !plat_is_dir(p)){ char old[1200]; snprintf(old,sizeof old,"%ssave/system",g_datadir);
        if(plat_is_dir(old)){ LOG("fs: copying %s to %s (save folders are now separated by type)\n",old,p); copy_tree(old,p); } }
      out_obj(c,hostfs_new(p)); return 0; }
    case 22: case 23: case 25: case 26: case 27: case 28: case 32: case 33: case 34: case 35: case 36: case 37: return 0;
    case 60: case 61: case 62: out_obj(c,svcobj_new("ISaveDataInfoReader",savereader,0)); return 0;
    case 200: { if(!rtotal) { WARN("fs: the game requests the RomFS but there is none (put the romfs folder in %s)\n",g_datadir); }
      out_obj(c,svcobj_new("IStorage(romfs)",romstorage,0)); return 0; }
    case 202: { u64 id=in64(c,8); SvcObj* syn=sysdata_open(id);
      if(syn){ LOG("fs: system data %016llx (synthetic copy)\n",(unsigned long long)id); out_obj(c,syn); return 0; }
      emptyrom_init(); LOG("fs: system data %016llx (not available on PC: provided empty)\n",(unsigned long long)id);
      out_obj(c,svcobj_new("IStorage(system)",emptystorage,0)); return 0; }
    case 203: return RC(2,1001);
    case 400: out_obj(c,svc_generic("IDeviceOperator")); return 0;
    case 500: out_obj(c,svc_generic("IEventNotifier")); return 0;
    case 1003: case 1004: case 1006: case 1007: case 1008: case 1011: case 1012: case 1013: return 0;
    case 1005: out32(c,0); return 0;
    case 1009: out_obj(c,svc_generic("IFileSystemProxyAccessLog")); return 0;
    case 620: case 630: case 631: case 632: case 800: return 0;
    case 8: case 11: case 17: case 19: case 31: return RC(2,1001);
    case 2: case 7: return RC(2,1001);
  }
  return stub_ok(o,c); }

static SvcObj* new_fsp(void){ return svcobj_new("fsp-srv",fsp_disp,0); }

static int fsppr_disp(SvcObj* o, IpcCtx* c){ (void)o; (void)c; return 0; }

static SvcObj* new_fsppr(void){ return svcobj_new("fsp-pr",fsppr_disp,0); }

void svc_fs_init(void){
  svc_register("fsp-srv",new_fsp); svc_register("fsp-pr",new_fsppr); svc_register("fsp-ldr",new_fsppr);
}
