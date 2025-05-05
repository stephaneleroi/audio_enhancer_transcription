#!/usr/bin/env python3
"""
Module de transcription audio optimisé pour les voix faibles.
"""

from typing import Any, Tuple, Dict
import logging
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import Audio

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
        logger.info(f"Sauvegarde des métadonnées dans {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("Sauvegarde JSON terminée")
    
    return text, metadata

def test_whisper_large_v3_french(audio_path, vad_segments=None):
    """
    Transcrit un fichier audio en utilisant le modèle bofenghuang/whisper-large-v3-french (HuggingFace Transformers) avec découpage automatique en segments de parole via pyannote.audio (CPU).
    Args:
        audio_path (str): Chemin du fichier audio à transcrire.
        vad_segments (list, optional): Liste de tuples (start, end) en secondes pour segmenter l'audio (optionnel, sinon VAD pyannote).
    Returns:
        Tuple[str, Dict]: (texte complet, métadonnées avec segments)
    """
    import torch
    import soundfile as sf
    import librosa
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import tempfile
    from pyannote.audio import Pipeline as PyannotePipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_name_or_path = "bofenghuang/whisper-large-v3-french"
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,  # pour les fichiers longs
        max_new_tokens=128,
    )
    metadata = {
        "segments": []
    }
    # Si pas de segments fournis, découpage automatique VAD pyannote
    if vad_segments is None:
        vad = PyannotePipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=None)
        vad_result = vad(audio_path)
        vad_segments = [(segment.start, segment.end) for segment in vad_result.get_timeline()]
    # Transcription segmentée
    audio, sr = librosa.load(audio_path, sr=16000)
    texts = []
    for i, (start, end) in enumerate(vad_segments):
        segment_audio = audio[int(start*sr):int(end*sr)]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_wav:
            sf.write(tmp_wav.name, segment_audio, sr)
            text = pipe(tmp_wav.name)["text"]
        texts.append(text)
        metadata["segments"].append({"id": i, "start": start, "end": end, "text": text})
    full_text = "\n\n".join(texts)
    return full_text, metadata

# --- Transcription WhisperX avec modèle officiel OpenAI ---
def transcribe_with_whisperx_large_v3(audio_path):
    """
    Transcrit un fichier audio avec WhisperX et le modèle officiel openai/whisper-large-v3.
    Args:
        audio_path (str): Chemin du fichier audio à transcrire.
    Returns:
        list: Liste de segments (dictionnaires avec start, end, text).
    """
    import whisperx
    model = whisperx.load_model("large-v3", device="cpu", compute_type="float32")
    result = model.transcribe(audio_path)
    return result["segments"]

# --- Transcription WhisperX avec modèle custom CTranslate2 ---
def transcribe_with_whisperx_custom_ctranslate2(audio_path, model_dir):
    """
    Transcrit un fichier audio avec WhisperX et un modèle custom converti au format CTranslate2.
    Args:
        audio_path (str): Chemin du fichier audio à transcrire.
        model_dir (str): Dossier du modèle CTranslate2 (contenant model.bin, config.json, etc.).
    Returns:
        list: Liste de segments (dictionnaires avec start, end, text).
    """
    import whisperx
    model = whisperx.load_model(model_dir, device="cpu", compute_type="float32")
    result = model.transcribe(audio_path)
    return result["segments"]

def print_whisperx_large_v3_segments(audio_path):
    """
    Transcrit et affiche les segments du fichier audio avec WhisperX (modèle officiel large-v3).
    Args:
        audio_path (str): Chemin du fichier audio à transcrire.
    """
    segments = transcribe_with_whisperx_large_v3(audio_path)
    for s in segments:
        print(f"[{s['start']:.2f}s - {s['end']:.2f}s] {s['text']}")
