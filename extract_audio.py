#!/usr/bin/env python3
"""
Script simple pour extraire l'audio d'un fichier vers WAV avec ffmpeg
"""

import os
import sys
import subprocess

def extract_audio_to_wav(input_file):
    """Extrait l'audio vers WAV avec ffmpeg."""
    
    if not os.path.exists(input_file):
        print(f"❌ Fichier non trouvé: {input_file}")
        return False
    
    # Nom du fichier de sortie
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}.wav"
    
    # Si le fichier WAV existe déjà, on le garde
    if os.path.exists(output_file):
        print(f"✅ Fichier WAV existe déjà: {output_file}")
        return True
    
    try:
        print(f"🎵 Conversion {input_file} → {output_file}...")
        
        # Commande ffmpeg pour conversion
        cmd = [
            'ffmpeg', '-i', input_file,
            '-acodec', 'pcm_s16le',  # Codec WAV standard
            '-ar', '16000',          # Fréquence d'échantillonnage 16kHz
            '-ac', '1',              # Mono
            '-y',                    # Overwrite output file
            output_file
        ]
        
        # Exécution de ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Audio extrait: {output_file}")
            if os.path.exists(output_file):
                size_mb = os.path.getsize(output_file) / 1024 / 1024
                print(f"   Taille: {size_mb:.1f} MB")
            return True
        else:
            print(f"❌ Erreur ffmpeg: {result.stderr}")
            return False
        
    except FileNotFoundError:
        print("❌ ffmpeg non trouvé. Veuillez l'installer.")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_audio.py <fichier_audio>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    success = extract_audio_to_wav(input_file)
    sys.exit(0 if success else 1) 