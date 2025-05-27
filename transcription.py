#!/usr/bin/env python3
"""
Module de transcription audio optimisé pour les voix faibles.
"""

from typing import Any, Tuple, Dict
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def transcribe_audio_improved(audio_path: str, model: Any = None, output_json=None) -> Tuple[str, Dict]:
    """
    Transcrit un fichier audio avec des paramètres optimisés pour les voix faibles.
    
    Args:
        audio_path: chemin vers le fichier audio
        model: modèle Whisper préchargé (optionnel)
        output_json: chemin pour sauvegarder les métadonnées (optionnel)
    """
    logger.info(f"Démarrage de la transcription de {audio_path}")
    logger.info(f"Output JSON configuré : {output_json}")
    
    try:
        import torch
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("Veuillez installer faster_whisper: pip install faster-whisper")
    
    if model is None:
        logger.info("Initialisation du modèle Whisper")
        # Utilisation du modèle large-v3 qui est plus précis pour les voix faibles
        model = WhisperModel(
            "large-v3", 
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
            cpu_threads=4,  # Optimisation multi-thread
            num_workers=2   # Workers pour le traitement parallèle
        )
    
    # Contexte initial enrichi pour mieux guider la transcription
    initial_prompt = """
    Transcription d'une réunion professionnelle avec plusieurs intervenants.
    Certains participants parlent à voix basse ou sont éloignés du microphone.
    Le vocabulaire utilisé est formel et technique.
    """
    
    logger.info("Démarrage de la transcription avec Whisper")
    # Configuration optimisée pour capter les voix faibles
    segments, info = model.transcribe(
        audio_path,
        language="fr",
        initial_prompt=initial_prompt,
        vad_filter=True,
        word_timestamps=True,
        beam_size=5,           # Augmentation de la recherche des hypothèses
        patience=2.0,          # Patience accrue pour les segments difficiles
        temperature=0.0,       # Pas de sampling aléatoire
        best_of=5,             # Sélectionner la meilleure transcription parmi 5
        condition_on_previous_text=True,  # Utiliser le contexte précédent
        vad_parameters=dict(
            min_silence_duration_ms=250,  # Détecter des silences plus courts
            speech_pad_ms=150,           # Ajouter plus de contexte
            threshold=0.30,              # Seuil plus bas pour détecter les voix faibles
        )
    )
    
    logger.info("Transcription terminée, traitement des segments")
    
    # Conversion des segments en liste pour éviter l'épuisement de l'itérateur
    segments_list = list(segments)
    logger.info(f"Nombre de segments détectés : {len(segments_list)}")
    
    if len(segments_list) == 0:
        logger.warning("Aucun segment détecté ! Vérification des paramètres VAD")
    
    # Extraction du texte des segments avec saut de ligne
    text = "\n\n".join([s.text for s in segments_list])  # Double saut de ligne entre segments
    logger.info(f"Longueur du texte extrait : {len(text)} caractères")
    logger.debug(f"Début du texte : {text[:200]}...")
    
    # Métadonnées enrichies
    metadata = {
        "duration": info.duration,
        "language": info.language,
        "segments": []  # Initialisation de la liste vide
    }
    
    logger.info(f"Durée détectée : {info.duration} secondes")
    logger.info(f"Langue détectée : {info.language}")
    
    # Ajout des segments dans les métadonnées
    for i, s in enumerate(segments_list):
        logger.debug(f"Traitement du segment {i}: {s.text[:50]}...")
        logger.debug(f"Temps: {s.start:.2f}s -> {s.end:.2f}s, Confiance: {s.avg_logprob}")
        
        segment_data = {
            "id": i,
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "confidence": s.avg_logprob,
        }
        
        # Ajout des mots si disponibles
        if hasattr(s, "words") and s.words:
            words = list(s.words)  # Conversion en liste pour éviter l'épuisement
            logger.debug(f"Nombre de mots dans le segment {i}: {len(words)}")
            segment_data["words"] = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.probability
                }
                for w in words
            ]
        
        metadata["segments"].append(segment_data)
    
    logger.info(f"Nombre total de segments traités : {len(metadata['segments'])}")
    
    # Écriture des métadonnées dans un fichier JSON si demandé
    if output_json:
        import json
        import os
        logger.info(f"Sauvegarde des métadonnées dans {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("Sauvegarde JSON terminée")
        
        # Génération automatique du fichier .txt
        txt_filename = os.path.splitext(output_json)[0] + '_transcript.txt'
        logger.info(f"Sauvegarde de la transcription texte dans {txt_filename}")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            # En-tête du fichier
            f.write(f"TRANSCRIPTION AUDIO\n")
            f.write(f"==================\n\n")
            f.write(f"Fichier source : {audio_path}\n")
            f.write(f"Durée : {info.duration:.2f} secondes\n")
            f.write(f"Langue détectée : {info.language}\n")
            f.write(f"Nombre de segments : {len(metadata['segments'])}\n")
            f.write(f"\n" + "="*50 + "\n\n")
            
            # Transcription complète
            f.write("TRANSCRIPTION COMPLÈTE :\n")
            f.write("-" * 25 + "\n\n")
            f.write(text)
            
            # Transcription avec timestamps (optionnel)
            f.write(f"\n\n" + "="*50 + "\n\n")
            f.write("TRANSCRIPTION AVEC TIMESTAMPS :\n")
            f.write("-" * 32 + "\n\n")
            
            for segment in metadata['segments']:
                start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
                end_time = f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}"
                confidence = segment['confidence']
                f.write(f"[{start_time} - {end_time}] (conf: {confidence:.3f})\n")
                f.write(f"{segment['text']}\n\n")
        
        logger.info("Sauvegarde TXT terminée")
    
    return text, metadata
