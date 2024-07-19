import os
import sys
import shutil
import torch
import subprocess
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
import whisper
import numpy as np
from pyannote.audio import Pipeline
from huggingface_hub import HfFolder
from config import CACHE_DIR, MODELS_DIR, PYANNOTE_AUTH_TOKEN

# Import and apply the patch
import patch_pyannote

# Set environment variables
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch')
os.environ['XDG_CACHE_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')

# Ensure cache directories exist
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)

# Set Hugging Face token
HfFolder.save_token(PYANNOTE_AUTH_TOKEN)

class TranscriptionThread(QThread):
    progress = pyqtSignal(int)
    transcription_chunk = pyqtSignal(str)
    transcription_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    speaker_labels_ready = pyqtSignal(list)  # New signal to emit when speaker labels are ready

    def __init__(self, file_path, model_name, output_file, use_diarization=True, auto_detect_speakers=True, num_speakers=None):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name
        self.output_file = output_file
        self.use_diarization = use_diarization
        self.auto_detect_speakers = auto_detect_speakers
        self.num_speakers = num_speakers
        self.is_cancelled = False
        self.device = self.get_device()
        self.speaker_names = {}  # Dictionary to store speaker names

    def get_device(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.FloatTensor(1)
                return torch.device("cuda")
            except RuntimeError as e:
                print(f"CUDA error: {e}")
                return torch.device("cpu")
        return torch.device("cpu")

    def log(self, message):
        print(f"TranscriptionThread: {message}")

    def convert_audio(self, input_file):
        output_file = os.path.join(CACHE_DIR, os.path.basename(input_file).rsplit('.', 1)[0] + ".wav")
        self.log(f"Converting {input_file} to {output_file}")
        try:
            subprocess.run(["ffmpeg", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", "-y", output_file], check=True)
            return output_file
        except subprocess.CalledProcessError as e:
            self.log(f"Error converting file: {e}")
            return input_file

    def run(self):
        try:
            self.log(f"Starting transcription for file: {self.file_path}")
            self.log(f"Using device: {self.device}")
            
            base_model_name = self.model_name.split()[0]
            
            model_path = os.path.join(MODELS_DIR, f"{base_model_name}.pt")
            self.log(f"Looking for model file at: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Whisper model file not found: {model_path}")

            self.log("Loading Whisper model")
            model = whisper.load_model(base_model_name, device=self.device, download_root=MODELS_DIR)
            
            self.log("Starting transcription")
            self.progress.emit(10)
            
            result = model.transcribe(self.file_path, fp16=(self.device.type == "cuda"))
            self.progress.emit(50)
            
            if self.use_diarization:
                self.log("Starting diarization")
                converted_file = self.convert_audio(self.file_path)

                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                    use_auth_token=PYANNOTE_AUTH_TOKEN)
                pipeline = pipeline.to(self.device)

                self.log("Diarization pipeline loaded")

                diarization_options = {}
                if not self.auto_detect_speakers and self.num_speakers is not None:
                    diarization_options['num_speakers'] = self.num_speakers

                self.log(f"Running diarization with options: {diarization_options}")
                diarization = pipeline(converted_file, **diarization_options)

                self.log("Diarization complete, combining results")
                self.log(f"Diarization output: {diarization}")

                transcription_with_speakers = self.combine_transcription_and_diarization(result["segments"], diarization)
                self.log(f"Combined transcription: {transcription_with_speakers[:5]}...")  # Log first 5 segments

                # Emit the unique speaker labels
                unique_speakers = list(set(segment['speaker'] for segment in transcription_with_speakers if 'speaker' in segment))
                self.speaker_labels_ready.emit(unique_speakers)
            else:
                self.log("Skipping diarization")
                transcription_with_speakers = [{'text': segment['text']} for segment in result["segments"]]

            self.log("Writing results to file")
            with open(self.output_file, 'w', encoding='utf-8') as f:
                current_speaker = None
                for segment in transcription_with_speakers:
                    if self.is_cancelled:
                        break
                    
                    if 'speaker' in segment and segment['speaker'] != current_speaker:
                        current_speaker = segment['speaker']
                        speaker_name = self.speaker_names.get(current_speaker, current_speaker)
                        f.write(f"\n[{speaker_name}]: ")
                    
                    f.write(segment['text'] + " ")
                    self.transcription_chunk.emit(segment['text'] + " ")
                    self.progress.emit(50 + int((transcription_with_speakers.index(segment) + 1) / len(transcription_with_speakers) * 50))

            if not self.is_cancelled:
                self.progress.emit(100)
                self.transcription_complete.emit()
                self.log("Transcription and diarization completed successfully")

            # Clean up temporary file
            if self.use_diarization and converted_file != self.file_path:
                try:
                    os.remove(converted_file)
                    self.log(f"Temporary file removed: {converted_file}")
                except FileNotFoundError:
                    self.log(f"Temporary file not found, skipping removal: {converted_file}")
                except Exception as e:
                    self.log(f"Error removing temporary file: {str(e)}")

        except Exception as e:
            self.log(f"Error occurred: {str(e)}")
            if not self.is_cancelled:
                self.error_occurred.emit(str(e))

    def combine_transcription_and_diarization(self, transcription_segments, diarization):
        combined = []
        current_speaker = None
        current_text = ""

        for segment in transcription_segments:
            if self.is_cancelled:
                break
            start = segment['start']
            end = segment['end']
            text = segment['text']
            
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if self.is_cancelled:
                    break
                if turn.start < end and turn.end > start:
                    speakers.append(speaker)
            
            if speakers:
                speaker = max(set(speakers), key=speakers.count)
            else:
                speaker = "Unknown"
            
            if speaker != current_speaker:
                if current_speaker is not None:
                    combined.append({
                        'speaker': current_speaker,
                        'text': current_text.strip()
                    })
                current_speaker = speaker
                current_text = text + " "
            else:
                current_text += text + " "
        
        if current_speaker is not None:
            combined.append({
                'speaker': current_speaker,
                'text': current_text.strip()
            })
        
        return combined

    def assign_speaker_names(self, names_dict):
        self.speaker_names = names_dict
        self.log(f"Speaker names assigned: {self.speaker_names}")

    def cancel(self):
        self.is_cancelled = True
        self.log("Transcription cancellation requested")

    @staticmethod
    def clear_cache():
        """Clear the cache directory."""
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
        os.makedirs(os.path.join(CACHE_DIR, 'torch'), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, 'transformers'), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, 'datasets'), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, 'huggingface'), exist_ok=True)