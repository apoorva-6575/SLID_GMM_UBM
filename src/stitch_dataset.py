"""
Script to stitch short audio files into exactly 30-second clips.
This is required to achieve the high accuracy benchmarked in the paper.
"""

import os
import sys
import shutil
import csv
from pathlib import Path
import soundfile as sf
import numpy as np

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 30.0
TARGET_FRAMES = int(TARGET_SR * TARGET_DURATION)

def read_metadata(csv_path):
    metadata = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    return metadata

def write_metadata(csv_path, metadata):
    if not metadata:
        return
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

def process_split(split, data_dir):
    print(f"\n--- Processing split: {split.upper()} ---")
    split_dir = Path(data_dir) / split
    audio_dir = split_dir / "audio"
    backup_dir = split_dir / "audio_original"
    meta_path = split_dir / "metadata.csv"
    meta_backup_path = split_dir / "metadata_original.csv"
    
    if not audio_dir.exists() and not backup_dir.exists():
        print(f"Skipping {split}, audio directory not found.")
        return
        
    # 1. Backup original files
    if not backup_dir.exists():
        print(f"Backing up {audio_dir} to {backup_dir}...")
        os.rename(audio_dir, backup_dir)
        os.makedirs(audio_dir)
    else:
        print(f"Backup already exists at {backup_dir}, skipping backup step.")
        os.makedirs(audio_dir, exist_ok=True)
        
    if not meta_backup_path.exists():
        print(f"Backing up metadata to {meta_backup_path}...")
        shutil.copy2(meta_path, meta_backup_path)
        
    # 2. Read metadata from backup
    metadata = read_metadata(meta_backup_path)
    
    # Group by language
    from collections import defaultdict
    lang_files = defaultdict(list)
    for row in metadata:
        lang_files[row['language']].append(row)
        
    new_metadata = []
    
    # 3. Stitch audio files per language
    for lang, rows in lang_files.items():
        print(f"  Stitching {lang} files...")
        buffer = []
        buffer_frames = 0
        clip_counter = 0
        
        for idx, row in enumerate(rows):
            orig_filepath = backup_dir / row['filepath']
            if not orig_filepath.exists():
                continue
                
            try:
                # Load audio
                y, sr = sf.read(orig_filepath)
                if len(y.shape) > 1:
                    y = y.mean(axis=1)
                
                buffer.append(y)
                buffer_frames += len(y)
                
                while buffer_frames >= TARGET_FRAMES:
                    concat_y = np.concatenate(buffer)
                    stitched_y = concat_y[:TARGET_FRAMES]
                    
                    remainder_y = concat_y[TARGET_FRAMES:]
                    buffer = [remainder_y] if len(remainder_y) > 0 else []
                    buffer_frames = len(remainder_y)
                    
                    new_filename = f"{lang}_stitched_{clip_counter:04d}.wav"
                    out_path = audio_dir / new_filename
                    sf.write(out_path, stitched_y, TARGET_SR)
                    
                    new_metadata.append({
                        'filepath': new_filename,
                        'language': lang,
                        'speaker_id': f"stitched_{lang}",
                        'duration': TARGET_DURATION,
                        'split': split
                    })
                    clip_counter += 1
            except Exception as e:
                pass
                
        print(f"    Created {clip_counter} 30-second clips for {lang}.")
        
    # 4. Write new metadata
    print(f"Saving new metadata for {split} with {len(new_metadata)} total clips...")
    write_metadata(meta_path, new_metadata)

def main():
    data_dir = "d:\\SLID_GMM_UBM\\data"
    for split in ['train', 'test', 'valid']:
        process_split(split, data_dir)
        
    print("\nStitching complete! All files in 'audio' are now exactly 30 seconds long.")
    print("Original files are preserved in 'audio_original' directories.")

if __name__ == "__main__":
    main()
