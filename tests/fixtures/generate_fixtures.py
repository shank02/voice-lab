"""Generate small WAV fixture files for testing."""

import csv
import os
import numpy as np
import soundfile as sf

FIXTURE_DIR = os.path.dirname(__file__)
CLIPS_DIR = os.path.join(FIXTURE_DIR, "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

SR = 16000
SAMPLES = [
    ("clip_001.mp3", "Hello world", 1.0),
    ("clip_002.mp3", "Testing one two three", 1.5),
    ("clip_003.mp3", "The quick brown fox", 2.0),
    ("clip_004.mp3", "Speech recognition test", 1.0),
    ("clip_005.mp3", "Good morning everyone", 1.5),
    ("clip_006.mp3", "", 1.0),  # Empty sentence — should be filtered
]

# Write WAV files (we'll use .mp3 names but WAV format for simplicity in tests)
for filename, sentence, duration_s in SAMPLES:
    n_samples = int(SR * duration_s)
    # Generate a simple sine wave
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    filepath = os.path.join(CLIPS_DIR, filename)
    sf.write(filepath, audio, SR, format="WAV")  # WAV even though named .mp3

# Write TSV
tsv_path = os.path.join(FIXTURE_DIR, "test.tsv")
with open(tsv_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["client_id", "path", "sentence", "up_votes", "down_votes"])
    for filename, sentence, _ in SAMPLES:
        writer.writerow(["client_1", filename, sentence, "2", "0"])

print(f"Generated {len(SAMPLES)} fixture clips in {CLIPS_DIR}")
print(f"Generated TSV at {tsv_path}")
