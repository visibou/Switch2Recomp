"""Find frame-to-frame glitches in a frame capture (SWR_CAPTURE, runtime Eden backend) and tie them to what Eden did.

  python tools/frame_check.py <run folder> [--strips 24] [--mp4]

Reads <run>/frames.bin (every composite, small, see renderer_vulkan.cpp SwrCapture), <run>/composites.csv (one row
per composite: present number, image address, channels parked when its tick was queued, whether it ran ahead of
earlier command lists, skipped GPU work, pipelines built) and <run>/frames.csv (one row per present). Writes:
  frame_check.txt   counts, rates per minute while moving, every hit with its composites.csv row, and how often each
                    log flag is set on hit frames compared with all frames (what the hits have in common)
  strips/           for each of the first --strips hits: frames N-3..N+1 side by side, N framed in red
  frames.mp4        (--mp4) the capture as a video at the capture's own pace
Kinds of hits (differences are mean absolute differences of 160x90 grey images, 0-255):
  back      frame N is much closer to N-2 or N-3 than to N-1: an older image shown again (the picture jumps back)
  flicker   N-1 and N+1 agree but N differs from both, and N is not an older frame
  repeat    N is the same image as N-1 while the frames around it move (a frame shown twice: a stutter)
  black     N is (nearly) black
  spike     the time from the composite before is over 2.5 times the median
Camera motion: the horizontal/vertical shift between N-1 and N (phase correlation), to see whether hits come with
turns (the user sees the twitching mostly when the camera turns).
"""
import argparse, csv, os, struct, subprocess, sys
import numpy as np

HDR = 32


def load_frames(path):
    with open(path, 'rb') as f: head = f.read(HDR)
    if len(head) < HDR or head[:4] != b'SWRF': sys.exit('%s: not a frame capture' % path)
    w, h = struct.unpack_from('<HH', head, 12)
    rec = HDR + w * h * 3
    n = os.path.getsize(path) // rec
    raw = np.memmap(path, dtype=np.uint8, mode='r', shape=(n, rec))
    meta = np.frombuffer(raw[:, :HDR].tobytes(), dtype=np.dtype([('magic', 'S4'), ('comp', '<u4'), ('present', '<u4'),
                                                                  ('w', '<u2'), ('h', '<u2'), ('t', '<u8'),
                                                                  ('pad', 'S8')]))
    pix = raw[:, HDR:].reshape(n, h, w, 3)
    return meta, pix, w, h


def grey_small(pix):
    """160x90 grey float images (block average), the inputs of every comparison."""
    n, h, w, _ = pix.shape
    out = np.empty((n, 90, 160), np.float32)
    fy, fx = h // 90, w // 160
    exact = h % 90 == 0 and w % 160 == 0
    ys, xs = np.linspace(0, h - 1, 90).astype(int), np.linspace(0, w - 1, 160).astype(int)
    for i in range(0, n, 256):
        if exact:   # block average
            g = pix[i:i + 256].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            out[i:i + 256] = g.reshape(-1, 90, fy, 160, fx).mean(axis=(2, 4))
        else:       # other sizes (256x144): nearest samples
            g = pix[i:i + 256][:, ys][:, :, xs].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            out[i:i + 256] = g
    return out


def diff(a, b): return float(np.abs(a - b).mean())


def shift(a, b):
    """(dx, dy) moving a onto b, phase correlation."""
    fa, fb = np.fft.rfft2(a - a.mean()), np.fft.rfft2(b - b.mean())
    r = fa.conj() * fb
    r /= np.abs(r) + 1e-6
    c = np.fft.irfft2(r, s=a.shape)
    y, x = np.unravel_index(int(np.argmax(c)), c.shape)
    if y > a.shape[0] // 2: y -= a.shape[0]
    if x > a.shape[1] // 2: x -= a.shape[1]
    return x, y


def read_csv(path):
    if not os.path.isfile(path): return []
    with open(path) as f: return list(csv.DictReader(f))


def save_png(path, img):
    from PIL import Image
    Image.fromarray(img).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--strips', type=int, default=24)
    ap.add_argument('--mp4', action='store_true')
    ap.add_argument('--noise', type=float, default=1.5, help='differences below this are the same image')
    a = ap.parse_args()
    run = a.run
    fb = os.path.join(run, 'frames.bin')
    if not os.path.isfile(fb): fb = os.path.join(run, 'captures', 'frames.bin')   # the game folder (capture my play)
    meta, pix, w, h = load_frames(fb)
    n = len(meta)
    comps = {int(r['n']): r for r in read_csv(os.path.join(run, 'composites.csv'))}
    # per composite: the image's modification tick the last time its address was shown before (image_unwritten)
    shown_tick, last_at = {}, {}
    for k in sorted(comps):
        r = comps[k]
        if 'fb_mod_tick' not in r: break
        shown_tick[r['n']] = last_at.get(r['address'], '')
        last_at[r['address']] = r['fb_mod_tick']
    g = grey_small(pix)
    d1 = np.zeros(n, np.float32); d2 = np.zeros(n, np.float32); d3 = np.zeros(n, np.float32)
    dn = np.zeros(n, np.float32)   # N-1 vs N+1
    mx = np.zeros(n, np.int32); my = np.zeros(n, np.int32)
    for i in range(1, n):
        d1[i] = diff(g[i], g[i - 1])
        if i >= 2: d2[i] = diff(g[i], g[i - 2])
        if i >= 3: d3[i] = diff(g[i], g[i - 3])
        if i + 1 < n: dn[i] = diff(g[i - 1], g[i + 1])
        mx[i], my[i] = shift(g[i - 1], g[i])
    t = meta['t'].astype(np.float64) / 1e9
    dt = np.diff(t, prepend=t[0]) * 1000.0
    med_dt = float(np.median(dt[1:])) if n > 1 else 0.0
    mean_luma = np.array([float(x.mean()) for x in g])
    # composites missing from the capture (the composite numbers jump): a hole would look like a jump forward
    comp_gap = np.diff(meta['comp'].astype(np.int64), prepend=meta['comp'][0])

    hits = []
    for i in range(3, n - 1):
        if comp_gap[i] != 1: continue   # a capture hole: no comparison across it
        older = min(d2[i], d3[i])
        if d1[i] > 3 * a.noise and older < 0.5 * d1[i] and d1[i + 1] > 3 * a.noise:
            hits.append((i, 'back'))
        elif d1[i] > 3 * a.noise and d1[i + 1] > 3 * a.noise and dn[i] < 0.5 * min(d1[i], d1[i + 1]):
            hits.append((i, 'flicker'))
        elif d1[i] < a.noise / 3 and d1[i - 1] > 3 * a.noise and d1[i + 1] > 3 * a.noise:
            hits.append((i, 'repeat'))
        if mean_luma[i] < 6 and mean_luma[i - 1] > 30: hits.append((i, 'black'))
        if med_dt and dt[i] > 2.5 * med_dt: hits.append((i, 'spike'))

    moving = d1 > 3 * a.noise
    minutes_moving = moving.sum() * (med_dt / 1000.0) / 60.0 if med_dt else 0.0
    kinds = ['back', 'flicker', 'repeat', 'black', 'spike']
    out = []
    out.append('%s: %d frames %dx%d, %.1f s (%.1f-%.1f s), %d capture holes, median composite interval %.1f ms' %
               (run, n, w, h, t[-1] - t[0], t[0], t[-1], int((comp_gap[1:] != 1).sum()), med_dt))
    out.append('moving frames: %d (%.2f min at the median interval)' % (moving.sum(), minutes_moving))
    for k in kinds:
        c = sum(1 for _, kk in hits if kk == k)
        out.append('  %-8s %5d   %.1f per minute moving' % (k, c, c / minutes_moving if minutes_moving else 0))

    # what the hits have in common: each composites.csv flag on hit frames vs all frames
    def flag_rates(idx):
        rows = [comps.get(int(meta['comp'][i])) for i in idx]
        rows = [r for r in rows if r]
        if not rows: return None
        def f(pred): return 100.0 * sum(1 for r in rows if pred(r)) / len(rows)
        return len(rows), {
            'parked': f(lambda r: int(r['parked_mask'], 16) != 0),
            'overtook': f(lambda r: r['overtook'] == '1'),
            'dispatch_skipped': f(lambda r: int(r['dispatch_skipped']) > 0),
            'work_failed': f(lambda r: int(r['native_work_failed']) > 0),
            'draw_no_pipeline': f(lambda r: int(r['draw_no_pipeline']) > 0),
            'draw_deferred': f(lambda r: int(r['draw_deferred']) > 0),
            'pipelines_built': f(lambda r: int(r['pipelines_built']) > 0),
            # the image not written since its address was last shown (old content; fb_* columns, later builds)
            'image_unwritten': f(lambda r: 'fb_mod_tick' in r and r['fb_mod_tick'] == shown_tick.get(r['n'], '')),
            'waited': f(lambda r: float(r.get('wait_ms') or 0) > 0),           # fresh composites: waited for it
            'dropped_before': f(lambda r: int(r.get('dropped') or 0) > 0),     # a composite dropped since the last
        }
    base = flag_rates([i for i in range(n) if moving[i]])
    if base:
        out.append('')
        out.append('composites.csv flags, %% of frames: all moving frames (%d) vs each kind of hit' % base[0])
        names = list(base[1])
        out.append('  %-18s %8s' % ('', 'moving') + ''.join('%9s' % k for k in kinds))
        per = {k: flag_rates([i for i, kk in hits if kk == k]) for k in kinds}
        for nm in names:
            out.append('  %-18s %7.1f%%' % (nm, base[1][nm]) +
                       ''.join(('%8.1f%%' % per[k][1][nm]) if per[k] else '        -' for k in kinds))
        out.append('  %-18s %8d' % ('frames', base[0]) + ''.join('%9d' % (per[k][0] if per[k] else 0) for k in kinds))

    out.append('')
    out.append('hits: frame  kind     t(s)    present comp   d(N,N-1) d(N,N-2) d(N,N-3) d(N-1,N+1)  shift   '
               'composites.csv (address parked overtook skipped failed nopipe deferred built)')
    for i, k in hits:
        c = int(meta['comp'][i]); r = comps.get(c, {})
        out.append('  %6d %-8s %7.2f %7d %6d   %6.1f   %6.1f   %6.1f   %6.1f   %+3d,%+3d   %s %s %s %s %s %s %s %s' % (
            i, k, t[i], meta['present'][i], c, d1[i], d2[i], d3[i], dn[i], mx[i], my[i],
            r.get('address', '-'), r.get('parked_mask', '-'), r.get('overtook', '-'), r.get('dispatch_skipped', '-'),
            r.get('native_work_failed', '-'), r.get('draw_no_pipeline', '-'), r.get('draw_deferred', '-'),
            r.get('pipelines_built', '-')))
    # presents the game made that never reached a composite, and presents composited twice
    pres = meta['present'].astype(np.int64)
    pd = np.diff(pres)
    out.append('')
    out.append('presents between captured composites: %s' % ', '.join('%d: %d' % (v, int((pd == v).sum()))
                                                                     for v in sorted(set(pd.tolist()))[:8]))

    # what spike frames did: the per-composite counters (work since the composite before) on spike frames vs the rest
    cols = [c for c in ('pipelines_built', 'tex_uploads', 'tex_upload_kib', 'tex_converted', 'tex_gpu_decoded',
                        'tex_upload_us', 'tex_staging_us', 'tex_read_us', 'tex_decode_us', 'bc_image_us', 'bc_decode_us', 'bc_staging_us', 'bc_encode_us', 'images_created', 'images_deleted', 'images_evicted', 'buffers_created',
                        'buffer_created_kib', 'buffer_upload_kib', 'gpu_busy_us', 'gpu_upload_us', 'gpu_transcode_us', 'gpu_astc_decode_us', 'image_join_us', 'image_alloc_us', 'astc_scan_us', 'transcode_cache_hits', 'transcode_cache_misses', 'burst_trimmed', 'burst_upgrades', 'wait_ms')
            if comps and c in next(iter(comps.values()))]
    def col(i, c):
        r = comps.get(int(meta['comp'][i]))
        try: return float(r[c]) if r else 0.0
        except (KeyError, ValueError): return 0.0
    spike_idx = [i for i in range(1, n) if med_dt and dt[i] > 2.5 * med_dt]
    calm_idx = [i for i in range(1, n) if med_dt and dt[i] <= 1.5 * med_dt]
    if cols and spike_idx:
        out.append('')
        out.append('per-composite counters, mean: frames over 2.5x the median interval (%d) vs frames under 1.5x (%d)' %
                   (len(spike_idx), len(calm_idx)))
        for c in cols:
            out.append('  %-20s %10.1f %10.1f' % (c, np.mean([col(i, c) for i in spike_idx]),
                                                  np.mean([col(i, c) for i in calm_idx]) if calm_idx else 0))

    # camera cuts: a frame far more different from the one before than the frames before it were from theirs
    cuts = []
    for i in range(8, n):
        if comp_gap[i] != 1: continue
        before = float(np.median(d1[i - 8:i]))
        if d1[i] > 25 and d1[i] > 3 * max(before, 2.0) and (not cuts or i - cuts[-1] > 8): cuts.append(i)
    if cuts:
        out.append('')
        worst = [max(dt[j] for j in range(i, min(n, i + 10))) for i in cuts]
        slow = [sum(1 for j in range(i, min(n, i + 10)) if dt[j] > 2 * med_dt) for i in cuts]
        out.append('camera cuts: %d; slowest frame in the 10 from the cut: median %.0f ms, max %.0f ms; frames over 2x '
                   'the median interval per cut: %.2f' % (len(cuts), np.median(worst), max(worst), np.mean(slow)))
        out.append('  frame    t(s)  slowest(ms) over2x  ' + ' '.join('%10s' % c[:10] for c in cols) + '   (sums over the 10 frames)')
        for i, wv, sv in zip(cuts, worst, slow):
            sums = [sum(col(j, c) for j in range(i, min(n, i + 10))) for c in cols]
            out.append('  %5d %7.2f %10.0f %6d  ' % (i, t[i], wv, sv) + ' '.join('%10.0f' % s for s in sums))
        if a.strips:
            sd = os.path.join(run, 'cuts'); os.makedirs(sd, exist_ok=True)
            for i in cuts[:a.strips]:   # the cut and the 4 frames after it, with their intervals
                tiles = [np.array(pix[j]) for j in range(i - 1, min(n, i + 5))]
                save_png(os.path.join(sd, '%05d_t%.2f_max%dms.png' % (i, t[i], max(dt[j] for j in range(i, min(n, i + 10))))),
                         np.concatenate(tiles, axis=1))
    text = '\n'.join(out) + '\n'
    open(os.path.join(run, 'frame_check.txt'), 'w').write(text)
    print(text[:6000])

    if a.strips and hits:
        sd = os.path.join(run, 'strips'); os.makedirs(sd, exist_ok=True)
        for i, k in hits[:a.strips]:
            tiles = []
            for j in range(i - 3, i + 2):
                im = np.array(pix[j])
                if j == i: im = im.copy(); im[:3] = im[-3:] = (255, 0, 0); im[:, :3] = im[:, -3:] = (255, 0, 0)
                tiles.append(im)
            save_png(os.path.join(sd, '%05d_%s_t%.2f.png' % (i, k, t[i])), np.concatenate(tiles, axis=1))
        print('strips:', sd)
    if a.mp4:
        # in real time: each 1/60 s shows the newest frame composited by then, so stalls play as long as they were
        mp4 = os.path.join(run, 'frames.mp4')
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (w, h),
                              '-r', '60', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', mp4],
                             stdin=subprocess.PIPE)
        j = 0
        for k in range(int((t[-1] - t[0]) * 60)):
            now = t[0] + k / 60.0
            while j + 1 < n and t[j + 1] <= now: j += 1
            p.stdin.write(np.ascontiguousarray(pix[j]).tobytes())
        p.stdin.close(); p.wait()
        print('video (real time):', mp4)


if __name__ == '__main__':
    main()
