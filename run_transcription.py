#!/usr/bin/env python3
import sys
from transcription import transcribe_audio_improved
import json

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_transcription.py <audio_file_path> [output_json_path]")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_json = None
    if len(sys.argv) > 2:
        output_json = sys.argv[2]
        print(f"Output JSON path: {output_json}")
    
    print(f"Transcribing {audio_file}...")
    try:
        # We pass output_json directly to the function if provided
        text, metadata = transcribe_audio_improved(audio_file, output_json=output_json)
        
        print("\n--- Transcription Text ---")
        print(text)
        
        print("\n--- Metadata ---")
        # Pretty print the metadata dictionary
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

        if output_json:
            print(f"\nMetadata also saved to {output_json}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1) 