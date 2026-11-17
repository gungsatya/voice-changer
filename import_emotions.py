"""Profile emotion clips from a folder (e.g. on Google Drive) without opening the UI.

    python import_emotions.py                      # uses $VOICE_DATA_DIR/emotion_samples
    python import_emotions.py /path/to/emotion_samples

Layout: <folder>/<Profile>/<emotion>.wav + <emotion>.txt (exact transcript).
Clips that were already profiled and have not changed are skipped, so running this at the start of
every Colab session is cheap.
"""

import sys

import voice_manager as vm

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Data folder: {vm.DATA_ROOT}")
    try:
        lines = vm.import_emotion_folder(folder)
    except ValueError as e:
        sys.exit(f"❌ {e}")
    print("\n".join(lines))
    for name in vm.list_profiles()[0]:
        print(f"  {name}: " + ", ".join(f"[{e}]" for e in vm.emotion_names(name)))
