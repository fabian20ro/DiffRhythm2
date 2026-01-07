#!/bin/bash
set -euo pipefail

#export PYTHONPATH=$PYTHONPATH:$PWD

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: ./generator.sh path/to/song.lrc [style prompt]

Inputs:
  path/to/song.lrc   Lyrics file; outputs are written next to this file.
  style prompt       Optional text prompt; if missing, a same-named .wav
                     next to the .lrc is used as the style prompt.

Expected assets:
  path/to/song.png   Optional; when present enables mp4 creation.

Outputs:
  path/to/song.jsonl
  path/to/song.mp3
  path/to/song.srt
  path/to/song.mp4
  path/to/song_final.mp4
EOF
  exit 0
fi

ffmpeg -version
espeak-ng --version

LYRICS_PATH=${1:-"results/test_song.lrc"}
STYLE_PROMPT=${2:-""}

RESULT_DIR="$(dirname "$LYRICS_PATH")"
SONG_NAME="$(basename "$LYRICS_PATH" .lrc)"
JSONL_PATH="$RESULT_DIR/$SONG_NAME.jsonl"
MP3_PATH="$RESULT_DIR/$SONG_NAME.mp3"
SRT_PATH="$RESULT_DIR/$SONG_NAME.srt"
ASS_PATH="$RESULT_DIR/$SONG_NAME.ass"
PNG_LOCATION="$RESULT_DIR/$SONG_NAME.png"
WAV_STYLE_PROMPT="$RESULT_DIR/$SONG_NAME.wav"
VIDEO_PATH="$RESULT_DIR/$SONG_NAME.mp4"
VIDEO_FINAL_PATH="$RESULT_DIR/${SONG_NAME}_final.mp4"

mkdir -p "$RESULT_DIR"

if [ ! -f "$MP3_PATH" ]; then
  if [ ! -f "$JSONL_PATH" ]; then
    if [ -f "$WAV_STYLE_PROMPT" ]; then
      STYLE_PROMPT="$WAV_STYLE_PROMPT"
      echo "using wav style prompt: $WAV_STYLE_PROMPT"
    fi
    if [ -z "$STYLE_PROMPT" ]; then
      echo "missing style prompt: provide arg or add $WAV_STYLE_PROMPT"
      exit 1
    fi
    echo "writing jsonl: $JSONL_PATH"
    python - "$JSONL_PATH" "$SONG_NAME" "$STYLE_PROMPT" "$LYRICS_PATH" <<'PY'
import json
import os
import sys

out_path, song_name, style_prompt, lyrics_path = sys.argv[1:5]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(json.dumps({
        "song_name": song_name,
        "style_prompt": style_prompt,
        "lyrics": lyrics_path,
    }) + "\n")
PY
  fi

  echo "generating mp3: $MP3_PATH"
  python inference.py \
    --repo-id ASLP-lab/DiffRhythm2 \
    --output-dir "$RESULT_DIR" \
    --input-jsonl "$JSONL_PATH" \
    --cfg-strength 0.8 \
    --steps 48 \
    --max-secs 420
else
  echo "mp3 exists, skipping: $MP3_PATH"
fi

if [ -f "$MP3_PATH" ] && [ ! -f "$ASS_PATH" ]; then
  echo "generating ass (karaoke): $ASS_PATH"
  python - "$MP3_PATH" "$ASS_PATH" <<'PY'
import sys
from faster_whisper import WhisperModel

audio_path, ass_path = sys.argv[1:3]
model = WhisperModel("medium")
segments, _info = model.transcribe(
    audio_path,
    language="en",
    word_timestamps=True,
)

def format_ass_timestamp(seconds):
    total_cs = int(round(seconds * 100.0))
    hours = total_cs // 360000
    total_cs %= 360000
    minutes = total_cs // 6000
    total_cs %= 6000
    secs = total_cs // 100
    cs = total_cs % 100
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"

def karaoke_text(words):
    parts = []
    for word in words:
        duration_cs = max(1, int(round((word.end - word.start) * 100.0)))
        text = word.word.replace("\n", " ").replace("\r", " ").strip()
        if not text:
            continue
        parts.append(r"{\k" + str(duration_cs) + "}" + text)
    return " ".join(parts).strip()

ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,0,2,80,80,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

with open(ass_path, "w") as f:
    f.write(ass_header)
    for segment in segments:
        start = format_ass_timestamp(segment.start)
        end = format_ass_timestamp(segment.end)
        words = getattr(segment, "words", None) or []
        if words:
            text = karaoke_text(words)
        else:
            text = (segment.text or "").replace("\n", " ").replace("\r", " ").strip()
        if not text:
            continue
        f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
PY
elif [ -f "$ASS_PATH" ]; then
  echo "ass exists, skipping: $ASS_PATH"
else
  echo "missing mp3, skipping ass: $MP3_PATH"
fi

if [ -f "$PNG_LOCATION" ]; then
  if [ -f "$MP3_PATH" ] && [ ! -f "$VIDEO_PATH" ]; then
    echo "generating mp4: $VIDEO_PATH"
    ffmpeg -loop 1 -i "$PNG_LOCATION" -i "$MP3_PATH" \
      -filter_complex "zoompan=z='min(zoom+0.0005,1.1)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'" \
      -c:a copy -c:v libx264 -shortest -pix_fmt yuv420p \
      "$VIDEO_PATH"
  elif [ -f "$VIDEO_PATH" ]; then
    echo "mp4 exists, skipping: $VIDEO_PATH"
  else
    echo "missing mp3, skipping mp4: $MP3_PATH"
  fi

  if [ -f "$VIDEO_PATH" ] && [ -f "$ASS_PATH" ] && [ ! -f "$VIDEO_FINAL_PATH" ]; then
    echo "burning subtitles: $VIDEO_FINAL_PATH"
    ffmpeg -i "$VIDEO_PATH" \
      -vf "ass=$ASS_PATH" \
      -c:a copy "$VIDEO_FINAL_PATH"
  elif [ -f "$VIDEO_FINAL_PATH" ]; then
    echo "final mp4 exists, skipping: $VIDEO_FINAL_PATH"
  else
    echo "missing mp4 or ass, skipping final mp4: $VIDEO_PATH / $ASS_PATH"
  fi
else
  echo "missing png, skipping video: $PNG_LOCATION"
fi
