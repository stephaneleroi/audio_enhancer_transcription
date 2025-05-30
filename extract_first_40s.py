#!/usr/bin/env python3
"""
Script pour extraire les 40 premières secondes d'un fichier audio
Usage: python extract_first_40s.py <fichier_audio>
"""

import os
import sys
from pathlib import Path
from pydub import AudioSegment

def extract_first_40_seconds(input_file):
    """Extrait les 40 premières secondes du fichier audio donné"""
    
    if not os.path.exists(input_file):
        print(f"❌ Fichier {input_file} non trouvé")
        return False
    
    # Génération automatique du nom de sortie
    input_path = Path(input_file)
    output_file = f"{input_path.stem}_40s{input_path.suffix}"
    
    try:
        print(f"🎵 Chargement de {input_file}...")
        
        # Détection automatique du format
        if input_path.suffix.lower() == '.mp3':
            audio = AudioSegment.from_mp3(input_file)
        elif input_path.suffix.lower() == '.wav':
            audio = AudioSegment.from_wav(input_file)
        elif input_path.suffix.lower() == '.m4a':
            audio = AudioSegment.from_file(input_file, format="m4a")
        else:
            # Tentative de détection automatique
            audio = AudioSegment.from_file(input_file)
        
        # Durée totale
        total_duration = len(audio) / 1000  # en secondes
        print(f"   Durée totale: {total_duration:.1f}s")
        
        # Vérification que le fichier fait au moins 40s
        if total_duration < 40:
            print(f"⚠️  Le fichier ne fait que {total_duration:.1f}s, extraction de la totalité")
            extract_duration = len(audio)
        else:
            extract_duration = 40 * 1000  # 40 secondes en millisecondes
        
        first_40s = audio[:extract_duration]
        
        print(f"✂️ Extraction des {len(first_40s) / 1000:.1f} premières secondes...")
        
        # Sauvegarde dans le même format que l'entrée
        print(f"💾 Sauvegarde vers {output_file}...")
        
        if input_path.suffix.lower() == '.mp3':
            first_40s.export(output_file, format="mp3")
        elif input_path.suffix.lower() == '.wav':
            first_40s.export(output_file, format="wav")
        elif input_path.suffix.lower() == '.m4a':
            first_40s.export(output_file, format="mp4")
        else:
            # Format par défaut
            first_40s.export(output_file, format="mp3")
        
        print(f"✅ Fichier {output_file} créé avec succès")
        print(f"   Taille: {os.path.getsize(output_file) / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_first_40s.py <fichier_audio>")
        print("Exemple: python extract_first_40s.py gros.mp3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    success = extract_first_40_seconds(input_file)
    sys.exit(0 if success else 1) 