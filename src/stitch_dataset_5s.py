import os
import csv
from pathlib import Path
import soundfile as sf
import numpy as np

# Configuration
TARGET_SR = 16000
CHUNK_DURATION = 5.0
CHUNK_FRAMES = int(TARGET_SR * CHUNK_DURATION)
TOTAL_DURATION = 30.0

def read_metadata(csv_path):
    metadata = []
    if not csv_path.exists():
        return metadata
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
    print(f"\n--- Processing 5-second chunks for split: {split.upper()} ---")
    split_dir = Path(data_dir) / split
    audio_30s_dir = split_dir / "audio"
    meta_30s_path = split_dir / "metadata.csv"
    
    audio_5s_dir = split_dir / "audio_5s"
    meta_5s_path = split_dir / "metadata_5s.csv"
    
    if not audio_30s_dir.exists() or not meta_30s_path.exists():
        print(f"Skipping {split}, 30-second audio or metadata not found.")
        return
        
    os.makedirs(audio_5s_dir, exist_ok=True)
    
    metadata_30s = read_metadata(meta_30s_path)
    metadata_5s = []
    
    print(f"Found {len(metadata_30s)} 30-second clips. Slicing into 5-second chunks...")
    
    for row in metadata_30s:
        orig_filename = row['filepath']
        lang = row['language']
        speaker = row['speaker_id']
        
        orig_filepath = audio_30s_dir / orig_filename
        if not orig_filepath.exists():
            continue
            
        try:
            # Load 30s audio
            y, sr = sf.read(orig_filepath)
            
            # If for some reason sr is wrong, skip or warn
            if sr != TARGET_SR:
                print(f"Warning: {orig_filename} has SR {sr}, skipping.")
                continue
                
            # Total frames should be ~480,000 (16000 * 30)
            # We slice into 6 chunks
            num_chunks = int(len(y) // CHUNK_FRAMES)
            if num_chunks > 6:
                num_chunks = 6 # Cap at 6 chunks per 30s file
                
            for i in range(num_chunks):
                start_frame = i * CHUNK_FRAMES
                end_frame = start_frame + CHUNK_FRAMES
                
                chunk_y = y[start_frame:end_frame]
                
                # Make sure it's exactly 5s (80000 frames)
                if len(chunk_y) == CHUNK_FRAMES:
                    new_filename = f"{orig_filename.replace('.wav', '')}_part{i+1}.wav"
                    out_path = audio_5s_dir / new_filename
                    sf.write(out_path, chunk_y, TARGET_SR)
                    
                    metadata_5s.append({
                        'filepath': new_filename,
                        'language': lang,
                        'speaker_id': f"{speaker}_part{i+1}",
                        'duration': CHUNK_DURATION,
                        'split': split
                    })
        except Exception as e:
            print(f"Error processing {orig_filename}: {e}")
            
    print(f"Created {len(metadata_5s)} 5-second clips for {split}.")
    print(f"Saving new metadata to {meta_5s_path}...")
    write_metadata(meta_5s_path, metadata_5s)

def main():
    data_dir = "d:\\SLID_GMM_UBM\\data"
    for split in ['train', 'test', 'valid']:
        process_split(split, data_dir)
        
    print("\n5-second stitching complete!")

if __name__ == "__main__":
    main()
