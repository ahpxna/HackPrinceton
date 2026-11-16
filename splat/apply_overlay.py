"""Verify and install this code overlay; retain original captures and checksums."""
import hashlib
from pathlib import Path
import re
import shutil
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def entries(path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            raise RuntimeError('Invalid checksum row in ' + str(path))
        sha, name = match.groups()
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or name in result:
            raise RuntimeError('Unsafe or duplicate manifest path: ' + name)
        result[name] = sha
    return result


def verify(root, rows):
    for name, sha in rows.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or digest(path) != sha:
            raise RuntimeError('Checksum/path check failed: ' + name)


def main():
    source = Path(__file__).resolve().parent
    target = Path(sys.argv[1]).expanduser().resolve()
    if source == target:
        raise RuntimeError('Run the installer from the separate overlay folder.')
    patch = entries(source / 'manifest.patch.sha256')
    verify(source, patch)
    manifest = target / 'manifest.sha256'
    original = entries(manifest)
    verify(target, original)
    # Include the patch manifest in the combined inventory without self-hashing.
    changed = {**patch, 'manifest.patch.sha256': digest(source / 'manifest.patch.sha256')}
    for name in changed:
        path = target / name
        if not path.resolve().is_relative_to(target) or path.is_symlink():
            raise RuntimeError('Refusing destination link: ' + name)
    files = set(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file())
    total = sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
    for name in changed:
        total += (source / name).stat().st_size - ((target / name).stat().st_size if name in files else 0)
    updated = {**original, **changed}
    text = ''.join(f'{sha}  {name}\n' for name, sha in sorted(updated.items()))
    total += len(text.encode()) - manifest.stat().st_size
    if total >= 2_200_000_000:
        raise RuntimeError(f'Pack would exceed 2.2 GB: {total} bytes')
    backup = target.parent / 'overlay_backups' / (time.strftime('%Y%m%d-%H%M%S') + '-' + digest(source / 'manifest.patch.sha256')[:8])
    backup.mkdir(parents=True, exist_ok=False)
    names = ['manifest.sha256'] + list(changed)
    created = []
    for name in names:
        old = target / name
        if old.exists():
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, saved)
        else:
            created.append(name)
    (backup / 'new-files.txt').write_text('\n'.join(created) + '\n')
    try:
        for name in changed:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.overlay-tmp')
            shutil.copy2(source / name, tmp)
            tmp.replace(dest)
        tmp = manifest.with_suffix('.tmp')
        tmp.write_text(text)
        tmp.replace(manifest)
        verify(target, updated)
    except BaseException:
        for name in names:
            old = backup / name
            if old.exists():
                shutil.copy2(old, target / name)
            elif name in created:
                (target / name).unlink(missing_ok=True)
        raise
    print(f'Overlay installed; {len(updated)} verified checksums; pack {total:,} bytes (<2.2 GB).')
    print('Original files saved outside the pack:', backup)
    print('Existing run roots have different source hashes: preserve them and use a new HP_SPLAT_ROOT.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 apply_overlay.py ~/hp_splat/pack')
    main()
