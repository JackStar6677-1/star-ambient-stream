#!/usr/bin/env python3
"""
audio_supervisor.py - Pipe estable de audio para FFmpeg.

Mantiene un motor de audio activo y permite reemplazarlo en caliente cuando
audio_engine.py o archivos de perfil cambian. El reemplazo entra con crossfade
para evitar que el directo vuelva a cero o haga cortes audibles.
"""

import array
import os
import signal
import struct
import subprocess
import sys
import time

SR = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2
ENGINE = "audio_engine.py"
WATCH_FILES = (
    ENGINE, "audio_profile.json", "audio_reload.flag",
    "generators/gen_base.py", "generators/gen_space.py", "generators/gen_polar.py",
    "generators/gen_nature.py", "generators/gen_submarine.py", "generators/gen_industrial.py",
)
CHUNK_SECONDS = 4
CHUNK_BYTES = SR * CHANNELS * SAMPLE_WIDTH * CHUNK_SECONDS
CROSSFADE_SECONDS = 28
CROSSFADE_CHUNKS = max(1, round(CROSSFADE_SECONDS / CHUNK_SECONDS))
CHECK_INTERVAL = 2.0

# Maximo de NaN consecutivos antes de loguear (evita spam)
_NAN_LOG_INTERVAL = 10
_nan_log_count = 0


def log(message, level="INFO"):
    print(f"[{level}] supervisor: {message}", file=sys.stderr, flush=True)


def watched_signature():
    """Firma liviana de archivos que deben activar cambio en caliente."""
    sig = []
    for path in WATCH_FILES:
        try:
            st = os.stat(path)
            sig.append((path, st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            sig.append((path, None, None))
        except Exception as exc:
            sig.append((path, "error", str(exc)))
    return tuple(sig)


def start_engine(label):
    log(f"iniciando motor {label}")
    return subprocess.Popen(
        [sys.executable, "-u", ENGINE],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        stdin=subprocess.DEVNULL,
        bufsize=0,
        start_new_session=True,
    )


def stop_engine(proc, label):
    if not proc:
        return
    if proc.poll() is not None:
        return
    log(f"deteniendo motor {label}")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=8)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def read_exact(proc):
    """Lee un chunk PCM completo. Devuelve None si el motor cae."""
    if proc.poll() is not None:
        return None
    data = bytearray()
    while len(data) < CHUNK_BYTES:
        part = proc.stdout.read(CHUNK_BYTES - len(data))
        if not part:
            return None
        data.extend(part)
    return bytes(data)


def sanitize_chunk(chunk):
    """Reemplaza muestras NaN/Inf/fuera de rango con silencio. Retorna bytes limpios."""
    global _nan_log_count
    samples = array.array("h")
    samples.frombytes(chunk)
    if sys.byteorder != "little":
        samples.byteswap()

    # Verificar si hay valores fuera de rango s16 (señal de NaN convertido a entero)
    # Los NaN de float32 al convertirse a int16 producen valores extremos (32767 o -32768 en bloque)
    n_bad = 0
    for i, s in enumerate(samples):
        if s == 32767 or s == -32768:
            n_bad += 1

    # Si mas del 50% del chunk son valores saturados, probablemente es NaN
    if n_bad > len(samples) * 0.5:
        _nan_log_count += 1
        if _nan_log_count % _NAN_LOG_INTERVAL == 1:
            log(f"chunk saturado detectado ({n_bad}/{len(samples)} muestras) - silenciando", "WARN")
        out = array.array("h", [0] * len(samples))
        if sys.byteorder != "little":
            out.byteswap()
        return out.tobytes()

    _nan_log_count = 0
    return chunk


def mix_crossfade(old_chunk, new_chunk, step, total):
    """Crossfade lineal entre dos chunks PCM s16le mono."""
    old_samples = array.array("h")
    new_samples = array.array("h")
    old_samples.frombytes(old_chunk)
    new_samples.frombytes(new_chunk)
    if sys.byteorder != "little":
        old_samples.byteswap()
        new_samples.byteswap()

    fade = min(1.0, max(0.0, (step + 1) / total))
    inv = 1.0 - fade
    out = array.array("h")
    out.extend(
        max(-32768, min(32767, int(a * inv + b * fade)))
        for a, b in zip(old_samples, new_samples)
    )
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def write_chunk(chunk):
    try:
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        raise


def crossfade_to_new_engine(current, reason):
    """Arranca un motor nuevo y cruza desde el actual."""
    replacement = start_engine("nuevo")
    wrote_any = False

    for step in range(CROSSFADE_CHUNKS):
        old_chunk = read_exact(current)
        new_chunk = read_exact(replacement)
        if new_chunk is None:
            log("el motor nuevo fallo durante crossfade; se conserva el actual", "WARN")
            stop_engine(replacement, "nuevo")
            return current, False
        if old_chunk is None:
            write_chunk(sanitize_chunk(new_chunk))
            wrote_any = True
            stop_engine(current, "anterior")
            log("motor anterior cayo; el nuevo queda activo")
            return replacement, True
        write_chunk(mix_crossfade(old_chunk, new_chunk, step, CROSSFADE_CHUNKS))
        wrote_any = True

    stop_engine(current, "anterior")
    log(f"hot-swap completado ({reason})")
    return replacement, wrote_any


def main():
    current = start_engine("inicial")
    signature = watched_signature()
    last_check = time.time()

    try:
        while True:
            now = time.time()
            if now - last_check >= CHECK_INTERVAL:
                new_signature = watched_signature()
                if new_signature != signature:
                    current, changed = crossfade_to_new_engine(current, "archivo modificado")
                    if changed:
                        signature = watched_signature()
                    last_check = now
                    continue
                last_check = now

            chunk = read_exact(current)
            if chunk is None:
                log("motor cayo; reiniciando sin cerrar pipe", "WARN")
                stop_engine(current, "fallido")
                current = start_engine("recuperacion")
                signature = watched_signature()
                continue
            write_chunk(sanitize_chunk(chunk))

    except BrokenPipeError:
        log("pipe cerrado por FFmpeg")
    except KeyboardInterrupt:
        log("detenido manualmente")
    finally:
        stop_engine(current, "actual")


if __name__ == "__main__":
    main()
