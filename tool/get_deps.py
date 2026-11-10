# SwitchRecomp - Copyright 2026 MZ Labs
# Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
# Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen.
# downloads what SwitchRecomp needs and doesn't come in the package, into the SwitchRecomp folder (no admin, no
# password, nothing installed on the system): GCC from conda-forge into gcc/, and on linux SDL2 into lib/ (from the
# pysdl2-dll wheel on PyPI). started by Install.bat (windows, after it got python) or Install.sh (linux).
#   get_deps.py           everything that's missing
#   get_deps.py gcc|sdl   only that
import hashlib, io, json, os, shutil, sys, tarfile, tempfile, time, urllib.request, zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tool', 'pylib'))   # zstandard for pythons older than 3.14
WIN = sys.platform.startswith('win')
URL = 'https://conda.anaconda.org/conda-forge/'
DEST = os.path.join(HERE, 'gcc')
CC = os.path.join(DEST, 'Library', 'bin', 'x86_64-w64-mingw32-gcc.exe') if WIN else os.path.join(DEST, 'bin', 'x86_64-conda-linux-gnu-gcc')

# package, sha256 (from conda-forge's index). gcc 16.2 for 64 bit windows (mingw-w64, ucrt) or for linux (glibc 2.34+)
WIN_PKGS = {
    'win-64/gcc_win-64-16.2.0-h5d11a75_2.conda': '11175ab515a5c7285d1cc91c48727dfb82785d7e8bcf8094b2600a0c415ecba8',
    'win-64/gcc_impl_win-64-16.2.0-h94e87e1_7.conda': '795d89968b7e67745360ff57ebc84dcae51dac2cbd4b4861be2523dbab97b71c',
    'win-64/binutils_win-64-2.46.1-default_hd1c8def_103.conda': '85a13f3a1d16b771cdf39715470789eb43981f2900ea5b4bbb0ef90b5301f2e3',
    'noarch/m2w64-sysroot_win-64-12.0.0.r4.gg4f2fc60ca-hd8ed1ab_10.conda': 'fb0ffe6b3c25189038c29abbd1fac2522d87fe2775a09e5f5088e5542dc3309b',
    'win-64/gendef-v12.0.0.r1.ggdc42231f0-he1bec0e_2.conda': 'ab5e10c99bff6ed9482c7293e2d8231d494756f81bebddfc42109c9d47c64385',
    'win-64/binutils_impl_win-64-2.46.1-default_ha84baeb_103.conda': '4f165824588d2600d202efc14727a5360f75508134a58b12943104f4731c696f',
    'win-64/libgcc-16.2.0-h110b43a_7.conda': '59feb6d16cba13a70f22dae70a44670eb2cc45e652e9a34a014814c2ca871132',
    'noarch/libgcc-devel_win-64-16.2.0-h254a5e0_107.conda': 'ff9a753514d9632ce9ab55e6872c875d15fea24514b999c9078df0151d6a26db',
    'win-64/libstdcxx-16.2.0-hae5796f_7.conda': '5984f81dcb289446fb3cdb0117c0680bded57eaa98da193f705400ec94fc71ae',
    'noarch/libstdcxx-devel_win-64-16.2.0-h230208c_107.conda': '1345b0c9187a7900370e2078b3568b6444d1c8de241f5d5a97e110fdc2a5f14a',
    'win-64/m2-conda-epoch-20250515-0_x86_64.conda': '51e9214548f177db9c3fe70424e3774c95bf19cd69e0e56e83abe2e393228ba1',
    'noarch/mingw-w64-ucrt-x86_64-crt-git-12.0.0.r4.gg4f2fc60ca-hd8ed1ab_10.conda': 'de3e42149b498c16bfb485b7729f4ca0fe392be576a2a10ff702d661799b1df3',
    'noarch/mingw-w64-ucrt-x86_64-headers-git-12.0.0.r4.gg4f2fc60ca-hd8ed1ab_10.conda': '1add86481f35163215e7076e6f06f22aa9f1f9345a5fff5cb07bc846c13fbec7',
    'noarch/mingw-w64-ucrt-x86_64-windows-default-manifest-6.4-he206cdd_7.conda': '5b0df4e0ba8487ffd59f60c34c5dbb9e001ecd2c5d2c66ba88eada40bfa3ecb8',
    'noarch/mingw-w64-ucrt-x86_64-winpthreads-git-12.0.0.r4.gg4f2fc60ca-hd8ed1ab_10.conda': '828abb111286940473c4c665fc8ab300d28920f5af83b32295e8bf2256a8f342',
    'win-64/libwinpthread-12.0.0.r4.gg4f2fc60ca-h57928b3_10.conda': '0fccf2d17026255b6e10ace1f191d0a2a18f2d65088fd02430be17c701f8ffe0',
    'win-64/ld_impl_win-64-2.46.1-default_hfd38196_103.conda': '4d84eaf3073ccc3935777e73d5508770c370d62279c1af261118cc91c5d55b73',
    'win-64/zstd-1.5.7-h534d264_7.conda': 'ca7daae4f218a11fab82cc2857f0ea518ec3f46acec60490485347a4c22c6b3e',
    'win-64/libzlib-1.3.2-hfd05255_3.conda': '0629c2cc0404d3bb29d6baa7b4ba62da80797015e86de050db81ea5a07050527',
}
LINUX_PKGS = {
    'noarch/sysroot_linux-64-2.34-h087de78_3.conda': '08c50c314b331730eeb8f87c39a45e5426a5463e078392ab283e2552d753443e',
    'linux-64/gcc_linux-64-16.2.0-h0ab548f_2.conda': 'b9817e310019a9c8b79e46335c87e478c7646d2fcc727cc53bb5a3e8ce276c1d',
    'noarch/tzdata-2026e-h5d301ff_0.conda': '2b7ca858879f183ce2dcc65b300f48a0133d5430550f35b23d80d405a0788dbf',
    'linux-64/gcc_impl_linux-64-16.2.0-h9891192_7.conda': '14acaa661e4c4ef6c65606612b8b08f8f079c6a9e6da3ffcb77056685c675651',
    'linux-64/binutils_linux-64-2.46.1-default_h4852527_103.conda': '606a42c754752fb905fe2012d52ff370d47b5b6adefd5f7befc4ec30ce758f23',
    'linux-64/binutils_impl_linux-64-2.46.1-default_hfdba357_103.conda': '98f9ddc6dc5d56a824b7eaea5bab6ad4bb47642288fc70d6b1b4024ac39d5595',
    'linux-64/libgcc-16.2.0-ha9f2e26_7.conda': 'e031634c3a928f9594eba868bc46ef5230d6571a364a0c63215b06c4563bfcea',
    'noarch/libgcc-devel_linux-64-16.2.0-he3ce08f_107.conda': 'f1e7465f6b53f89b325e441c30ad7b19407b051eca556b6d0386ec05513b8e76',
    'linux-64/libsanitizer-16.2.0-h3048135_7.conda': 'e383569c331239308a8663facb76208fc0e141131ff3d17c1c2e50f34716a5a9',
    'linux-64/libstdcxx-16.2.0-h934c35e_7.conda': 'fa7018298629fc429971fffd6ecba3d27a7b02e4e890d6a93ea01ffebe8cb745',
    'noarch/libstdcxx-devel_linux-64-16.2.0-h86e191b_107.conda': 'ebbad84bdebd8a9964020d74dcce0f211f1431878c66326f6115dcb59e4c3469',
    'linux-64/ld_impl_linux-64-2.46.1-default_hbd61a6d_103.conda': '901649b1edbe841fd4ee7e2d491a05184ed540eb01cdae506872f344babc1d28',
    'linux-64/zstd-1.5.7-hb78ec9c_7.conda': '47d682b9f6d6ec9eb1a6e6c3e75ea6273e899e78fb7fc59f81d39745009fbc60',
    'linux-64/libzlib-1.3.2-h25fd6f3_3.conda': 'eb8a0db0aa570124f7d2a93d7c7f596e3390df5e047818d873baad32985fc736',
    'noarch/kernel-headers_linux-64-5.14.0-he073ed8_3.conda': 'c4d973dd80ab10028b3e4c58c1c6d8f437bfeaf6069dafb1130dabc550e9caf4',
}
PKGS = WIN_PKGS if WIN else LINUX_PKGS


# SDL2 for the linux tool: pysdl2-dll 2.32.10 (manylinux2014, runs on glibc 2.17+)
SDL_WHEEL = ('https://files.pythonhosted.org/packages/5d/6e/07ec92ba241b2aa941abf478e96c1474f2c5975b61d84303fbcb2ba1dcc9/'
             'pysdl2_dll-2.32.10-py2.py3-none-manylinux2014_x86_64.whl',
             '12b136c38114ae0f8a85453a879795ba5d61663d98673ba19c36c3008ff00353')
LIB = os.path.join(HERE, 'lib')


def fetch(url, sha, cache, tries=4):
    if os.path.isfile(cache) and (not sha or hashlib.sha256(open(cache, 'rb').read()).hexdigest() == sha):
        return open(cache, 'rb').read()
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            if sha and hashlib.sha256(data).hexdigest() != sha:
                raise IOError('the download is damaged (checksum)')
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            open(cache, 'wb').write(data)
            return data
        except Exception as e:
            if t == tries - 1: raise
            print('  %s, trying again...' % e); time.sleep(2 * (t + 1))


def zstd_module():
    # python 3.14 has zstd built in. older ones (the system's python on linux) get the zstandard wheel from PyPI
    # for exactly this python, unpacked into tool/pylib
    try:
        from compression import zstd
        return lambda d: zstd.decompress(d)
    except ImportError:
        pass
    try:
        import zstandard
    except ImportError:
        tag = 'cp%d%d' % sys.version_info[:2]
        plat = 'win_amd64' if WIN else 'x86_64'
        info = json.load(urllib.request.urlopen('https://pypi.org/pypi/zstandard/json', timeout=60))
        whl = [u for u in info['urls'] if '-%s-%s-' % (tag, tag) in u['filename'] and u['filename'].endswith('.whl')
               and plat in u['filename'] and (WIN or 'manylinux' in u['filename'])]
        if not whl: raise RuntimeError('no zstandard for Python %d.%d, install Python 3.14 or newer' % sys.version_info[:2])
        u = whl[0]
        data = fetch(u['url'], u['digests']['sha256'], os.path.join(HERE, 'tool', 'pylib', u['filename']))
        pylib = os.path.join(HERE, 'tool', 'pylib')
        zipfile.ZipFile(io.BytesIO(data)).extractall(pylib)
        import importlib
        sys.path_importer_cache.pop(pylib, None); importlib.invalidate_caches()
        import zstandard
    return lambda d: zstandard.ZstdDecompressor().stream_reader(io.BytesIO(d)).read()


_unzstd = None
def unzstd(data):
    global _unzstd
    if not _unzstd: _unzstd = zstd_module()
    return _unzstd(data)


def get(path, sha, tries=4):
    return fetch(URL + path, sha, os.path.join(DEST, '_downloads', path.split('/')[-1]), tries)


def relocate(paths, prefix):
    # conda packages carry a placeholder where the install folder goes: text files get the real path, binaries
    # get it padded with zeros to the same length
    for p in paths:
        ph = p.get('prefix_placeholder')
        if not ph: continue
        f = os.path.join(DEST, p['_path'])
        if not os.path.isfile(f) or os.path.islink(f): continue
        data = open(f, 'rb').read(); old = ph.encode()
        if old not in data: continue
        if p.get('file_mode') == 'binary':
            new = prefix.encode()
            if len(new) > len(old): continue
            out = bytearray()
            i = 0
            while True:
                j = data.find(old, i)
                if j < 0: out += data[i:]; break
                k = data.find(b'\0', j)
                if k < 0: k = len(data)
                s = data[j:k].replace(old, new)
                out += data[i:j] + s + b'\0' * (k - j - len(s)); i = k
            data = bytes(out)
        else:
            data = data.replace(old, (prefix.replace('\\', '/') if WIN and b'/' in old else prefix).encode())
        mode = os.stat(f).st_mode
        os.chmod(f, mode | 0o200); open(f, 'wb').write(data); os.chmod(f, mode)


def unpack(data):
    z = zipfile.ZipFile(io.BytesIO(data))
    paths = []
    for n in z.namelist():
        if not n.endswith('.tar.zst'): continue
        t = tarfile.open(fileobj=io.BytesIO(unzstd(z.read(n))))
        if n.startswith('info-'):
            try: paths = json.load(t.extractfile('info/paths.json'))['paths']
            except Exception: paths = []
            continue
        for m in t.getmembers():
            if m.name.startswith('/') or '..' in m.name.split('/'): continue
            out = os.path.join(DEST, m.name)
            if m.isdir(): os.makedirs(out, exist_ok=True); continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.lexists(out): os.remove(out)
            if m.issym():
                if WIN: continue
                os.symlink(m.linkname, out)
            elif m.islnk():
                src = os.path.join(DEST, m.linkname)
                try: os.link(src, out)
                except OSError: shutil.copyfile(src, out)
            elif m.isfile():
                with t.extractfile(m) as s, open(out, 'wb') as d: shutil.copyfileobj(s, d)
                os.chmod(out, (m.mode & 0o777) | 0o600)
    return paths


def works():
    import subprocess
    if not os.path.isfile(CC): return False
    d = tempfile.mkdtemp()
    try:
        c = os.path.join(d, 't.c'); open(c, 'w').write('int swr_test(int x){ return x*3; }\n')
        o = os.path.join(d, 't.dll' if WIN else 't.so')
        env = dict(os.environ)
        if WIN: env['PATH'] = os.path.join(DEST, 'Library', 'bin') + os.pathsep + env.get('PATH', '')
        r = subprocess.run([CC, '-O2', '-shared', '-fPIC', c, '-o', o], capture_output=True, text=True, env=env)
        if r.returncode: print(r.stdout + r.stderr)
        return r.returncode == 0 and os.path.isfile(o)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def sdl_ok():
    return all(os.path.isfile(os.path.join(LIB, n)) for n in ('libSDL2-2.0.so.0', 'libSDL2_ttf-2.0.so.0', 'libSDL2_image-2.0.so.0'))


def install_sdl():
    print('SDL2 (the window and text of the SwitchRecomp tool)')
    if sdl_ok():
        print('  already there\n'); return 0
    try:
        data = fetch(SDL_WHEEL[0], SDL_WHEEL[1], os.path.join(LIB, '_downloads', SDL_WHEEL[0].split('/')[-1]))
    except Exception as e:
        print('  the download failed: %s' % e); return 1
    z = zipfile.ZipFile(io.BytesIO(data))
    os.makedirs(os.path.join(LIB, 'licenses'), exist_ok=True)
    # the main three under their sonames, the rest (image formats SDL_image opens by itself) as they are
    names = {'libSDL2-2.0.so': 'libSDL2-2.0.so.0', 'libSDL2_ttf-2.0.so': 'libSDL2_ttf-2.0.so.0', 'libSDL2_image-2.0.so': 'libSDL2_image-2.0.so.0'}
    keep = ('libtiff.so.5', 'libwebp.so.1.0.3', 'libwebpdemux.so.2.6.0', 'libavif.so.16', 'libdav1d.so.6')
    for n in z.namelist():
        b = n.split('/')[-1]
        if n.startswith('sdl2dll/dll/') and (b in names or b in keep):
            out = os.path.join(LIB, names.get(b, b))
            open(out, 'wb').write(z.read(n)); os.chmod(out, 0o755)
        elif '/licenses/' in n and not n.endswith('/'):
            out = os.path.join(LIB, 'licenses', n.split('/licenses/', 1)[1])
            os.makedirs(os.path.dirname(out), exist_ok=True); open(out, 'wb').write(z.read(n))
    for link, tgt in (('libwebp.so.7', 'libwebp.so.1.0.3'), ('libwebpdemux.so.2', 'libwebpdemux.so.2.6.0')):
        p = os.path.join(LIB, link)
        if os.path.lexists(p): os.remove(p)
        os.symlink(tgt, p)
    shutil.rmtree(os.path.join(LIB, '_downloads'), ignore_errors=True)
    print('  done\n' if sdl_ok() else '  something went wrong\n')
    return 0 if sdl_ok() else 1


def system_gcc():
    # linux: a gcc the system already has is enough, SwitchRecomp uses it when gcc/ isn't there
    return not WIN and shutil.which('gcc') is not None


def install_gcc(force=False):
    print('GCC (the compiler that builds the recompiled games)')
    if works():
        print('  already there\n'); return 0
    if not force and system_gcc():
        print('  the system has gcc (%s), using that one\n' % shutil.which('gcc')); return 0
    print('  downloading GCC 16.2 into %s (about %d MB, once)' % (DEST, 110 if WIN else 175))
    os.makedirs(DEST, exist_ok=True)
    total = len(PKGS)
    try:
        for i, (p, sha) in enumerate(PKGS.items()):
            print('  [%2d/%d] %s' % (i + 1, total, p.split('/')[-1])); sys.stdout.flush()
            relocate(unpack(get(p, sha)), DEST)
    except Exception as e:
        print('\n  the download failed: %s' % e)
        print('  check the internet connection and run it again (what was downloaded is kept)')
        return 1
    if not works():
        print('\n  GCC was downloaded but it does not compile. Send this window to the SwitchRecomp Discord.'); return 1
    shutil.rmtree(os.path.join(DEST, '_downloads'), ignore_errors=True)
    print('  done\n')
    return 0


def main():
    what = sys.argv[1:] or (['gcc'] if WIN else ['sdl', 'gcc'])
    print('SwitchRecomp: installing what is missing (into the SwitchRecomp folder, nothing goes on the system)\n')
    r = 0
    if 'sdl' in what and not WIN: r |= install_sdl()
    if 'gcc' in what: r |= install_gcc(force=os.environ.get('SWR_OWN_GCC') == '1')
    print('All done!' if not r else 'Something failed, see above.')
    return r


if __name__ == '__main__':
    sys.exit(main())
