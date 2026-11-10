"""nxnative's shader_split.py, with its zstd frames decoded in-process: it calls a `zstd` command for frames without a dictionary, which this PC does not have.
Run by split_mkw.sh, arguments as shader_split.py's. The patch sits at module level so the pool's workers (spawned, they import this file again) get it too."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import s2r_paths as P
sys.path.insert(0, P.ext('pylibs'))
sys.path.insert(0, P.ext('nxnative', 'dm', 'gfx', 'tools'))
import shader_split as S


def unzs(path):
    d = S.open(path, 'rb').read()
    did = S.zs_dict_id(d)
    if not did: return S.zstd_mod().decompress(d)
    return S.unzs_dict(d, did, path)


S.unzs = unzs
for name in ('cra_jobs', 'trpfs_jobs', 'pkz_jobs', 'cpk_jobs', 'arh_jobs', 'frostbite_jobs', 'smash_arc_jobs', 'kpka_jobs', 'zip_jobs', 'ue_legacy_jobs', 'bdt_jobs'):
    setattr(S, name, lambda files, *a, **k: ([], set()))

if __name__ == '__main__':
    sys.argv[0] = 'shader_split.py'
    import nso
    try: S.main()
    except nso.Error as e: S.red(str(e))
