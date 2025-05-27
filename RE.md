# AudioClarify - Transcription optimisée pour voix faibles

Un système complet d'amélioration audio et de transcription optimisé pour les enregistrements de réunions avec des voix faibles ou inaudibles.

## 📋 Description

AudioClarify est une solution développée pour résoudre le problème courant des voix faibles dans les enregistrements de réunions. Il combine deux technologies principales :

1. **Prétraitement audio avancé** : Amplifie et clarifie les voix faibles tout en préservant la qualité des voix normales.
2. **Transcription optimisée** : Utilise le modèle Whisper avec des paramètres spécialement configurés pour maximiser la précision des voix faibles.

Contrairement aux solutions de transcription standards qui échouent souvent lorsque des locuteurs sont éloignés du microphone ou parlent à voix basse, AudioClarify permet d'obtenir des transcriptions complètes et précises même dans des conditions d'enregistrement difficiles.

## ✨ Fonctionnalités

- 🔊 **Amélioration audio multi-niveaux** détectant et amplifiant sélectivement les voix faibles
- 🎯 **Réduction de bruit ciblée** préservant les informations vocales
- 📝 **Transcription optimisée** utilisant le modèle Whisper large-v3
- ⏱️ **Horodatage précis au niveau des mots**
- 📊 **Métadonnées détaillées** (niveau de confiance, timestamps)
- 🔧 **Interface en ligne de commande hautement configurable**
- 📄 **Formats de sortie multiples** (texte brut, JSON détaillé)

## 🚀 Installation

### Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)
- Carte graphique NVIDIA avec CUDA (recommandé mais non obligatoire)

### Étapes d'installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votreusername/audioclarify.git
   cd audioclarify
   ```

2. Créez un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Utilisation

### Commande de base

Pour traiter un fichier audio et obtenir une transcription :

```bash
python audio_enhancer.py --input chemin/vers/audio.wav --transcribe
```

### Options principales

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--input`, `-i` | Chemin vers le fichier audio à traiter | (Requis) |
| `--output`, `-o` | Chemin de sortie pour l'audio amélioré | `input_enhanced.wav` |
| `--transcribe`, `-tr` | Activer la transcription après traitement | False |
| `--export-json`, `-j` | Exporter les métadonnées de transcription en JSON | False |
| `--verbose`, `-d` | Afficher les messages de débogage | False |

### Options d'amélioration audio avancées

| Option | Description | Plage | Défaut |
|--------|-------------|-------|--------|
| `--voice-boost`, `-v` | Amplification des voix faibles | 1.0-10.0 | 3.5 |
| `--ultra-boost`, `-u` | Amplification des voix très faibles | 1.0-15.0 | 5.0 |
| `--threshold`, `-t` | Seuil en dB pour les voix faibles | -35 à -15 | -28 |
| `--noise-reduction`, `-n` | Facteur de réduction du bruit | 0.5-3.0 | 1.8 |
| `--compression`, `-c` | Ratio de compression | 1.0-8.0 | 3.0 |

### Exemples d'utilisation

Transcription simple avec paramètres par défaut :
```bash
python audio_enhancer.py -i reunion.wav --transcribe
```

Amélioration audio seule avec paramètres personnalisés :
```bash
python audio_enhancer.py -i reunion.wav -o reunion_amelioree.wav --voice-boost 4.0 --noise-reduction 2.0
```

Traitement complet avec export des métadonnées :
```bash
python audio_enhancer.py -i reunion.wav --transcribe --export-json --voice-boost 4.5 --ultra-boost 6.0 --threshold -30
```

## 🔧 Structure du projet

```
audioclarify/
├── audio_enhancer.py   # Module principal d'amélioration audio
├── transcription.py    # Module de transcription optimisée
├── requirements.txt    # Dépendances du projet
├── README.md           # Documentation du projet
└── examples/           # Exemples d'utilisation
```

## 🧪 Fonctionnement technique

### Prétraitement audio

Le module d'amélioration audio applique plusieurs techniques de traitement du signal :

1. **Soustraction spectrale** pour la réduction du bruit de fond
2. **Filtrage passe-bande** pour isoler les fréquences vocales (300-3400 Hz)
3. **Amplification multi-niveaux** selon l'intensité des signaux
4. **Compression dynamique** pour réduire l'écart entre sons forts et faibles
5. **Amélioration de la clarté vocale** avec un focus accru sur les voix faibles
6. **Normalisation** pour éviter la saturation

### Transcription optimisée

La transcription utilise le modèle Whisper via l'implémentation `faster-whisper` avec :

1. **Modèle large-v3** pour une précision maximale
2. **Contextualisation enrichie** via un prompt initial spécifique
3. **Détection vocale améliorée** avec seuil réduit pour capter les voix faibles
4. **Paramètres d'inférence optimisés** (taille de faisceau, patience, etc.)
5. **Horodatage des mots** pour un alignement précis texte-audio

## 📚 Exemples de résultats

| Type d'audio | Sans traitement | Avec AudioClarify |
|--------------|----------------|-------------------|
| Réunion avec voix faibles | ~60% de précision | ~85-95% de précision |
| Enregistrement avec bruit | Nombreuses omissions | Transcription complète |
| Questions du public | Souvent manquantes | Captées avec précision |

## 📝 Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## ✅ Prérequis système recommandés

- **Processeur** : Intel Core i5/i7 ou équivalent AMD
- **RAM** : 8 Go minimum, 16 Go recommandés
- **GPU** : NVIDIA avec support CUDA (fortement recommandé)
- **Stockage** : Dépend de la taille des fichiers audio, minimum 1 Go d'espace libre

## 🛠️ Dépendances principales

- faster-whisper
- librosa
- soundfile
- scipy
- numpy
- torch

## 📆 Feuille de route

- [ ] Identification des locuteurs (diarisation)
- [ ] Interface graphique
- [ ] Support de formats audio supplémentaires
- [ ] Traitement par lots
- [ ] API Web

## 👥 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## 📞 Support

Pour toute question ou problème, veuillez ouvrir une issue sur GitHub.
