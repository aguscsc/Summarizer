# Summarizer
A tool that summarizes audio into text, with a focus on academic use, anotate key points of a class, summarize really long lectures, or record yourself reading the material to then have it summarized.

## Updates I'd like to make
- Have it to ask questions to you about the material
- direct mic transcription
- phone compatibility

# Requirements 
- whisper (for managing audio)
- Mistral (for managing text)
- Torch

# Example
For now it only manages to recieve audio files and then craft a brief summary, this example was made by feeding a video explaining ohm's law (https://youtu.be/_rSHqvjDksg?si=K-6AEUJeF_q6pKgO)
![example](pics/first.png)

# Installation

## Ollama
- linux
```
curl -fsSL https://ollama.com/install.sh | sh
```
- Windows & MacOS
visit Ollama's website (https://ollama.com/download/windows) / (https://ollama.com/download/mac)

## Whisper
```
sudo pacman -S python-openai-whisper
```
## Running summarizer
```
git clone https://github.com/aguscsc/Summarizer.git
cd code
python summarizer.py
```
