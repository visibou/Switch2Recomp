/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* AppImage runtime for the linux package. the squashfs with the whole package goes right after this ELF (the AppImage
   type 2 layout, so --appimage-extract and the usual tools work on it). the tool writes its settings, shader caches
   and the games it builds next to itself and a mounted AppImage is read only, so instead of mounting it we unpack it
   once into ~/.local/share/SwitchRecomp (SWITCHRECOMP_HOME for another folder) and run it from there. a newer
   AppImage unpacks over it: package files get replaced, games and settings stay. the squashfs is read with
   squashfuse's library (BSD license, see THIRD_PARTY.txt) */
#define _GNU_SOURCE
#include <squashfuse/squashfuse.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define APP_NAME "SwitchRecomp"

/* squashfuse is built with LZO support, but liblzo2 is GPL and our image is zstd. this keeps it out: an LZO image
   just fails to read */
int lzo1x_decompress_safe(const unsigned char* src, unsigned long srclen, unsigned char* dst, unsigned long* dstlen, void* wrk){
  (void)src; (void)srclen; (void)dst; (void)dstlen; (void)wrk; return -1; }

/* where the ELF ends: the squashfs starts there */
static long elf_end(const char* path){
  FILE* f=fopen(path,"rb"); if(!f) return -1;
  Elf64_Ehdr eh; long end=-1;
  if(fread(&eh,sizeof eh,1,f)==1 && !memcmp(eh.e_ident,ELFMAG,SELFMAG) && eh.e_ident[EI_CLASS]==ELFCLASS64){
    end=(long)(eh.e_shoff+(Elf64_Off)eh.e_shnum*eh.e_shentsize);
    for(int i=0;i<eh.e_phnum;i++){ Elf64_Phdr ph;
      if(fseek(f,(long)(eh.e_phoff+(Elf64_Off)i*eh.e_phentsize),SEEK_SET) || fread(&ph,sizeof ph,1,f)!=1) break;
      if((long)(ph.p_offset+ph.p_filesz)>end) end=(long)(ph.p_offset+ph.p_filesz); } }
  fclose(f); return end; }

static int mkdirs(const char* p){
  char b[PATH_MAX]; snprintf(b,sizeof b,"%s",p);
  for(char* s=b+1;*s;s++) if(*s=='/'){ *s=0; if(mkdir(b,0755) && errno!=EEXIST) return -1; *s='/'; }
  return mkdir(b,0755) && errno!=EEXIST? -1 : 0; }

/* files go to a temp name first and get renamed over the old one, so a copy that's still running keeps its file */
static int put_file(sqfs* fs, sqfs_inode* in, const char* dst, int mode){
  char tmp[PATH_MAX+8]; snprintf(tmp,sizeof tmp,"%s.new",dst);
  int fd=open(tmp,O_WRONLY|O_CREAT|O_TRUNC,mode? mode : 0644); if(fd<0) return -1;
  static char buf[1<<20]; sqfs_off_t off=0, size=(sqfs_off_t)in->xtra.reg.file_size; int ok=1;
  while(off<size && ok){ sqfs_off_t n= size-off>(sqfs_off_t)sizeof buf? (sqfs_off_t)sizeof buf : size-off;
    if(sqfs_read_range(fs,in,off,&n,buf)!=SQFS_OK || n<=0){ ok=0; break; }
    for(sqfs_off_t w=0;w<n;){ ssize_t k=write(fd,buf+w,(size_t)(n-w)); if(k<=0){ ok=0; break; } w+=k; }
    off+=n; }
  if(close(fd)) ok=0;
  if(ok){ chmod(tmp,mode); if(rename(tmp,dst)) ok=0; }
  if(!ok) unlink(tmp);
  return ok? 0 : -1; }

static int unpack(const char* img, long off, const char* dest, int verbose){
  sqfs fs; if(sqfs_open_image(&fs,img,(size_t)off)!=SQFS_OK){ fprintf(stderr,APP_NAME ": can't read the image inside %s\n",img); return -1; }
  if(mkdirs(dest)){ fprintf(stderr,APP_NAME ": can't create %s (%s)\n",dest,strerror(errno)); sqfs_destroy(&fs); return -1; }
  sqfs_traverse t; sqfs_err err=SQFS_OK; int bad=0, n=0;
  if(sqfs_traverse_open(&t,&fs,sqfs_inode_root(&fs))!=SQFS_OK){ sqfs_destroy(&fs); return -1; }
  while(sqfs_traverse_next(&t,&err)){
    if(t.dir_end) continue;
    sqfs_inode in; if(sqfs_inode_get(&fs,&in,t.entry.inode)!=SQFS_OK){ bad++; continue; }
    char p[PATH_MAX]; if(snprintf(p,sizeof p,"%s/%s",dest,t.path)>=(int)sizeof p){ bad++; continue; }
    int mode=in.base.mode&07777;
    switch(in.base.inode_type){
      case SQUASHFS_DIR_TYPE: case SQUASHFS_LDIR_TYPE:
        if(mkdir(p,(mode|0700)&07777) && errno!=EEXIST) bad++;
        break;
      case SQUASHFS_REG_TYPE: case SQUASHFS_LREG_TYPE:
        if(put_file(&fs,&in,p,mode)) bad++; else if(verbose) printf("%s\n",p);
        n++; break;
      case SQUASHFS_SYMLINK_TYPE: case SQUASHFS_LSYMLINK_TYPE: {
        char tgt[PATH_MAX]; size_t sz=sizeof tgt;
        if(sqfs_readlink(&fs,&in,tgt,&sz)!=SQFS_OK){ bad++; break; }
        unlink(p); if(symlink(tgt,p)) bad++; break; }
      default: break; } }
  sqfs_traverse_close(&t); sqfs_destroy(&fs);
  if(err!=SQFS_OK || bad){ fprintf(stderr,APP_NAME ": %d file(s) could not be unpacked into %s\n",bad? bad : 1,dest); return -1; }
  return n; }

/* the squashfs superblock's build time and size tell one AppImage from another */
static void image_id(const char* img, long off, char* out, size_t n){
  unsigned char sb[96]; FILE* f=fopen(img,"rb"); out[0]=0; if(!f) return;
  if(!fseek(f,off,SEEK_SET) && fread(sb,1,sizeof sb,f)==sizeof sb){ unsigned t, used_lo, used_hi;
    memcpy(&t,sb+8,4); memcpy(&used_lo,sb+40,4); memcpy(&used_hi,sb+44,4);
    snprintf(out,n,"%08x-%08x%08x",t,used_hi,used_lo); }
  fclose(f); }

int main(int argc, char** argv){
  char self[PATH_MAX]; ssize_t sl=readlink("/proc/self/exe",self,sizeof self-1); if(sl<=0){ fprintf(stderr,APP_NAME ": can't find myself\n"); return 1; } self[sl]=0;
  long off=elf_end(self); if(off<=0){ fprintf(stderr,APP_NAME ": %s has no image inside\n",self); return 1; }
  if(argc>1 && !strcmp(argv[1],"--appimage-offset")){ printf("%ld\n",off); return 0; }
  if(argc>1 && !strcmp(argv[1],"--appimage-extract")){ return unpack(self,off,"squashfs-root",1)<0; }
  if(argc>1 && (!strcmp(argv[1],"--appimage-help") || !strcmp(argv[1],"--appimage-version"))){
    printf(APP_NAME " AppImage\n  --appimage-extract   unpack into ./squashfs-root\n  --appimage-offset    where the squashfs starts\n"
           "It unpacks itself into ~/.local/share/" APP_NAME " (or $SWITCHRECOMP_HOME) and runs from there.\n"); return 0; }

  char dest[PATH_MAX]; const char* h=getenv("SWITCHRECOMP_HOME");
  if(h && *h) snprintf(dest,sizeof dest,"%s",h);
  else { const char* x=getenv("XDG_DATA_HOME"); const char* home=getenv("HOME");
    if(x && *x) snprintf(dest,sizeof dest,"%s/" APP_NAME,x);
    else if(home && *home) snprintf(dest,sizeof dest,"%s/.local/share/" APP_NAME,home);
    else { fprintf(stderr,APP_NAME ": no HOME, set SWITCHRECOMP_HOME to a folder to unpack into\n"); return 1; } }

  char id[64], have[64]={0}, idp[PATH_MAX+32]; image_id(self,off,id,sizeof id); snprintf(idp,sizeof idp,"%s/.appimage_id",dest);
  { FILE* f=fopen(idp,"r"); if(f){ if(!fgets(have,sizeof have,f)) have[0]=0; fclose(f); have[strcspn(have,"\r\n")]=0; } }
  char run[PATH_MAX+16]; snprintf(run,sizeof run,"%s/AppRun",dest);
  if(!id[0] || strcmp(id,have) || access(run,X_OK)){
    fprintf(stderr,APP_NAME ": unpacking into %s (first run of this version)...\n",dest);
    int n=unpack(self,off,dest,0); if(n<0) return 1;
    FILE* f=fopen(idp,"w"); if(f){ fprintf(f,"%s\n",id); fclose(f); }
    fprintf(stderr,APP_NAME ": %d files ready\n",n); }

  setenv("APPIMAGE",self,1); setenv("APPDIR",dest,1); setenv("ARGV0",argv[0],1);
  argv[0]=run; execv(run,argv);
  fprintf(stderr,APP_NAME ": can't run %s (%s)\n",run,strerror(errno)); return 1; }
