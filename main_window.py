import os
import time
import sys
import json
import requests
from PyQt6.QtWidgets import (QMainWindow, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, QHBoxLayout,
                             QWidget, QProgressBar, QLabel, QComboBox, QMessageBox, QMenuBar, QMenu,
                             QCheckBox, QSizePolicy, QApplication, QDialog, QLineEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
import torch

from transcription_thread import TranscriptionThread
from model_manager import ModelManagerDialog, ModelUtils
from config import WHISPER_MODELS, MODELS_DIR, PYANNOTE_AUTH_TOKEN

class SpeakerNamingDialog(QWidget):
    namesUpdated = pyqtSignal(dict)

    def __init__(self, speaker_labels, parent=None):
        super().__init__(parent, Qt.WindowType.Window)  # Make it a separate window
        self.setWindowTitle("Name Speakers")
        self.setGeometry(300, 300, 400, 300)
        self.speaker_labels = speaker_labels
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Create table
        self.table = QTableWidget(len(self.speaker_labels), 2)
        self.table.setHorizontalHeaderLabels(["Speaker Label", "Custom Name"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Populate table
        for i, label in enumerate(self.speaker_labels):
            self.table.setItem(i, 0, QTableWidgetItem(label))
            self.table.setItem(i, 1, QTableWidgetItem(""))

        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply Names")
        apply_button.clicked.connect(self.apply_names)
        button_layout.addWidget(apply_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connect cellChanged signal to apply_names method
        self.table.cellChanged.connect(self.apply_names)

    def apply_names(self):
        speaker_names = {}
        for i in range(self.table.rowCount()):
            label = self.table.item(i, 0).text()
            name = self.table.item(i, 1).text()
            if name:
                speaker_names[label] = name
        self.namesUpdated.emit(speaker_names)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Transcription App with Speaker Diarization")
        self.setGeometry(100, 100, 700, 500)

        self.input_file_path = None
        self.output_file_path = None
        self.transcription_thread = None
        self.start_time = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_elapsed_time)
        self.speaker_naming_dialog = None
        self.speaker_names = {}

        # LLM settings
        self.llm_base_url = None
        self.llm_model = None

        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize UI elements
        self.setup_menu()
        self.setup_ui()

        # Update GPU status after UI is set up
        self.update_gpu_status()

    def setup_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        select_input_action = QAction('Select Input File', self)
        select_input_action.triggered.connect(self.select_input_file)
        file_menu.addAction(select_input_action)
        
        select_output_action = QAction('Select Output File', self)
        select_output_action.triggered.connect(self.select_output_file)
        file_menu.addAction(select_output_action)

        quit_application_action = QAction('Exit', self)
        quit_application_action.triggered.connect(self.quit_application)
        file_menu.addAction(quit_application_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        manage_models_action = QAction('Manage Models', self)
        manage_models_action.triggered.connect(self.open_model_manager)
        tools_menu.addAction(manage_models_action)
        
        clear_cache_action = QAction('Clear Cache', self)
        clear_cache_action.triggered.connect(self.clear_cache)
        tools_menu.addAction(clear_cache_action)

        # LLM menu
        llm_menu = menubar.addMenu('LLM')
        
        llm_settings_action = QAction('LLM Settings', self)
        llm_settings_action.triggered.connect(self.open_llm_settings)
        llm_menu.addAction(llm_settings_action)

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Model selection section
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Whisper Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.update_model_combo()
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        main_layout.addLayout(model_layout)

        # File selection section
        file_layout = QHBoxLayout()
        self.file_button = QPushButton("Select Input File")
        self.file_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.file_button.clicked.connect(self.select_input_file)
        file_layout.addWidget(self.file_button)
        self.file_label = QLabel("No input file selected")
        file_layout.addWidget(self.file_label)
        main_layout.addLayout(file_layout)

        # File length label
        self.file_length_label = QLabel("File Length: N/A")
        main_layout.addWidget(self.file_length_label)

        # Output file selection section
        output_layout = QHBoxLayout()
        self.output_button = QPushButton("Select Output File")
        self.output_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.output_button.clicked.connect(self.select_output_file)
        output_layout.addWidget(self.output_button)
        self.output_label = QLabel("No output file selected")
        output_layout.addWidget(self.output_label)
        main_layout.addLayout(output_layout)

        # Speaker detection options
        speaker_layout = QHBoxLayout()
        self.use_diarization_checkbox = QCheckBox("Use Speaker Diarization")
        self.use_diarization_checkbox.setChecked(False)
        speaker_layout.addWidget(self.use_diarization_checkbox)
        
        self.auto_detect_speakers_checkbox = QCheckBox("Auto-detect Number of Speakers")
        self.auto_detect_speakers_checkbox.setChecked(True)
        self.auto_detect_speakers_checkbox.setEnabled(False)
        speaker_layout.addWidget(self.auto_detect_speakers_checkbox)
        
        self.num_speakers_combo = QComboBox()
        self.num_speakers_combo.addItems([str(i) for i in range(1, 11)])
        self.num_speakers_combo.setCurrentIndex(1)  # Default to 2 speakers
        self.num_speakers_combo.setEnabled(False)
        self.num_speakers_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        speaker_layout.addWidget(QLabel("Number of Speakers:"))
        speaker_layout.addWidget(self.num_speakers_combo)
        speaker_layout.addStretch()
        
        main_layout.addLayout(speaker_layout)

        # Connect checkboxes
        self.use_diarization_checkbox.stateChanged.connect(self.update_speaker_options)
        self.auto_detect_speakers_checkbox.stateChanged.connect(self.update_speaker_options)

        # Transcribe, Cancel, and Summarize buttons
        button_layout = QHBoxLayout()
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setEnabled(False)
        button_layout.addWidget(self.transcribe_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.cancel_button.clicked.connect(self.cancel_transcription)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setStyleSheet("background-color: #f44336;")
        button_layout.addWidget(self.cancel_button)

        self.summarize_button = QPushButton("Summarize")
        self.summarize_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.summarize_button.clicked.connect(self.summarize_transcript)
        self.summarize_button.setEnabled(False)
        button_layout.addWidget(self.summarize_button)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Progress bar and label
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("0%")
        progress_layout.addWidget(self.progress_label)
        main_layout.addLayout(progress_layout)

        # Time information
        time_layout = QHBoxLayout()
        self.start_time_label = QLabel("Start Time: Not started")
        time_layout.addWidget(self.start_time_label)
        self.end_time_label = QLabel("End Time: Not finished")
        time_layout.addWidget(self.end_time_label)
        self.elapsed_time_label = QLabel("Elapsed Time: 00:00:00")
        time_layout.addWidget(self.elapsed_time_label)
        main_layout.addLayout(time_layout)

        # Text edit for transcription output
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        main_layout.addWidget(self.text_edit)

        # GPU availability label
        self.gpu_label = QLabel()
        self.gpu_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.gpu_label.mousePressEvent = self.toggle_gpu_info
        main_layout.addWidget(self.gpu_label)

        # Detailed GPU info (hidden by default)
        self.detailed_gpu_info = QLabel()
        self.detailed_gpu_info.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.detailed_gpu_info.hide()
        main_layout.addWidget(self.detailed_gpu_info)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def update_speaker_options(self):
        use_diarization = self.use_diarization_checkbox.isChecked()
        auto_detect = self.auto_detect_speakers_checkbox.isChecked()
        
        self.auto_detect_speakers_checkbox.setEnabled(use_diarization)
        self.num_speakers_combo.setEnabled(use_diarization and not auto_detect)

    def get_file_dialog(self, dialog_type, caption, directory, filter):
        options = QFileDialog.Option.DontUseNativeDialog
        if sys.platform == 'win32':
            # Use native file dialog on Windows
            options = QFileDialog.Option(0)  # This is equivalent to no options
        
        if dialog_type == 'open':
            return QFileDialog.getOpenFileName(self, caption, directory, filter, options=options)
        elif dialog_type == 'save':
            return QFileDialog.getSaveFileName(self, caption, directory, filter, options=options)

    def select_input_file(self):
        self.input_file_path, _ = self.get_file_dialog('open', "Select Audio/Video File", "", "Audio/Video Files (*.mp3 *.mp4 *.wav *.avi)")
        if self.input_file_path:
            self.file_label.setText(f"Selected: {os.path.basename(self.input_file_path)}")
            self.set_default_output_file()
            self.update_transcribe_button()
            self.update_file_length()

    def select_output_file(self):
        if self.input_file_path:
            default_name = os.path.splitext(os.path.basename(self.input_file_path))[0] + ".txt"
            default_path = os.path.join(self.output_dir, default_name)
        else:
            default_path = self.output_dir

        self.output_file_path, _ = self.get_file_dialog('save', "Save Transcription File", default_path, "Text Files (*.txt)")
        if self.output_file_path:
            self.output_label.setText(f"Save to: {os.path.basename(self.output_file_path)}")
            self.update_transcribe_button()

    def update_file_length(self):
        if self.input_file_path:
            try:
                import ffmpeg
                probe = ffmpeg.probe(self.input_file_path)
                duration = float(probe['streams'][0]['duration'])
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                self.file_length_label.setText(f"File Length: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            except Exception as e:
                self.file_length_label.setText(f"File Length: Unable to determine")
                print(f"Error determining file length: {str(e)}")
        else:
            self.file_length_label.setText("File Length: N/A")

    def set_default_output_file(self):
        if self.input_file_path:
            input_filename = os.path.basename(self.input_file_path)
            output_filename = os.path.splitext(input_filename)[0] + ".txt"
            self.output_file_path = os.path.join(self.output_dir, output_filename)
            self.output_label.setText(f"Save to: {output_filename}")

    def update_transcribe_button(self):
        self.transcribe_button.setEnabled(bool(self.input_file_path and self.output_file_path))

    def start_transcription(self):
        if not self.input_file_path or not self.output_file_path:
            return

        print(f"Attempting to transcribe file: {os.path.abspath(self.input_file_path)}")

        if not os.path.exists(self.input_file_path):
            QMessageBox.critical(self, "Error", f"Input file not found: {self.input_file_path}")
            return

        if not os.access(self.input_file_path, os.R_OK):
            QMessageBox.critical(self, "Error", f"Cannot read the input file: {self.input_file_path}")
            return

        file_size = os.path.getsize(self.input_file_path)
        if file_size == 0:
            QMessageBox.critical(self, "Error", f"The input file is empty: {self.input_file_path}")
            return

        print(f"File exists, is readable, and has size: {file_size} bytes")

        self.text_edit.clear()
        self.transcribe_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")

        selected_model = self.model_combo.currentText()
        if not self.ensure_model_downloaded(selected_model):
            QMessageBox.critical(self, "Error", "Model download cancelled or failed.")
            self.reset_ui()
            return

        # Check for pyannote.audio authentication token
        if not PYANNOTE_AUTH_TOKEN or PYANNOTE_AUTH_TOKEN == "your_auth_token_here":
            QMessageBox.critical(self, "Error", "Please set your pyannote.audio authentication token in the config file.")
            self.reset_ui()
            return

        self.start_time = time.time()
        self.start_time_label.setText(f"Start Time: {time.strftime('%H:%M:%S')}")
        self.end_time_label.setText("End Time: Not finished")
        self.elapsed_time_label.setText("Elapsed Time: 00:00:00")
        self.timer.start(1000)  # Update every second

        use_diarization = self.use_diarization_checkbox.isChecked()
        auto_detect_speakers = self.auto_detect_speakers_checkbox.isChecked()
        num_speakers = int(self.num_speakers_combo.currentText()) if not auto_detect_speakers else None

        self.transcription_thread = TranscriptionThread(
            self.input_file_path, selected_model, self.output_file_path,
            use_diarization=use_diarization,
            auto_detect_speakers=auto_detect_speakers,
            num_speakers=num_speakers
        )
        self.transcription_thread.progress.connect(self.update_progress)
        self.transcription_thread.transcription_chunk.connect(self.append_transcription)
        self.transcription_thread.transcription_complete.connect(self.finalize_transcription)
        self.transcription_thread.error_occurred.connect(self.handle_error)
        self.transcription_thread.speaker_labels_ready.connect(self.handle_speaker_labels)
        self.transcription_thread.start()

        # Update GPU status after starting transcription
        self.update_gpu_status()
    
    def append_transcription(self, text):
        for label, name in self.speaker_names.items():
            text = text.replace(f"[{label}]:", f"[{name}]:")
        self.text_edit.append(text)

    def handle_speaker_labels(self, speaker_labels):
        self.speaker_naming_dialog = SpeakerNamingDialog(speaker_labels, self)
        self.speaker_naming_dialog.namesUpdated.connect(self.update_speaker_names)
        self.speaker_naming_dialog.show()

    def update_speaker_names(self, names):
        self.speaker_names = names
        self.transcription_thread.assign_speaker_names(self.speaker_names)
        self.update_transcript_file()
        self.update_transcript_display()

    def update_transcript_display(self):
        self.text_edit.clear()
        with open(self.output_file_path, 'r', encoding='utf-8') as f:
            self.text_edit.setPlainText(f.read())
    
    def update_transcript_file(self):
        if not self.output_file_path:
            return

        with open(self.output_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(self.output_file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                for label, name in self.speaker_names.items():
                    line = line.replace(f"[{label}]:", f"[{name}]:")
                file.write(line)

    def cancel_transcription(self):
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.cancel()
            self.transcription_thread.wait()
            self.reset_ui()
            self.timer.stop()
            QMessageBox.information(self, "Transcription Cancelled", "The transcription process has been cancelled.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"{value}%")

    def append_transcription(self, text):
        self.text_edit.append(text)

    def finalize_transcription(self):
        self.reset_ui()
        end_time = time.time()
        self.end_time_label.setText(f"End Time: {time.strftime('%H:%M:%S')}")
        total_time = end_time - self.start_time
        self.elapsed_time_label.setText(f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        self.timer.stop()
        self.update_transcript_display()  # Update the display with final speaker names
        QMessageBox.information(self, "Transcription Complete", "The transcription has been saved successfully.")
        self.save_final_transcript()

    def save_final_transcript(self):
            final_output_path = self.output_file_path.replace('.txt', '_final.txt')
            with open(self.output_file_path, 'r', encoding='utf-8') as input_file, \
                open(final_output_path, 'w', encoding='utf-8') as output_file:
                for line in input_file:
                    for label, name in self.speaker_names.items():
                        line = line.replace(f"[{label}]:", f"[{name}]:")
                    output_file.write(line)
            QMessageBox.information(self, "Final Transcript Saved", f"The final transcript with custom speaker names has been saved to {final_output_path}")

    def handle_error(self, error_message):
        self.reset_ui()
        self.timer.stop()
        if "Speaker diarization model access error" in error_message:
            QMessageBox.critical(self, "Authentication Error", 
                                 error_message + "\n\nPlease update the PYANNOTE_AUTH_TOKEN in config.py and restart the application.")
        elif "A required privilege is not held by the client" in error_message:
            QMessageBox.critical(self, "Permission Error", 
                                 "A permission error occurred. Try clearing the cache from the Tools menu and run the transcription again.")
        else:
            QMessageBox.critical(self, "Error", f"An error occurred during transcription: {error_message}")

    def reset_ui(self):
        self.transcribe_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.file_button.setEnabled(True)
        self.output_button.setEnabled(True)
        self.model_combo.setEnabled(True)

    def open_model_manager(self):
        dialog = ModelManagerDialog(self)
        dialog.exec()
        # Update the model combo box after managing models
        self.update_model_combo()

    def update_model_combo(self):
        current_model = self.model_combo.currentText()
        self.model_combo.clear()
        for model in WHISPER_MODELS.keys():
            for language in WHISPER_MODELS[model].keys():
                model_name = f"{model} ({language})"
                if ModelUtils.is_model_installed(model, language):
                    self.model_combo.addItem(model_name)
        if current_model in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(current_model)
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)

    def ensure_model_downloaded(self, model_name):
        base_model_name = model_name.split()[0]  # Extract base model name
        model_path = os.path.join(MODELS_DIR, f"{base_model_name}.pt")

        if not os.path.exists(model_path):
            reply = QMessageBox.question(self, "Model Not Found", 
                                        f"The {base_model_name} model is not found. Do you want to download it?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                # Use ModelManagerDialog to download the model
                dialog = ModelManagerDialog(self)
                dialog.download_model(base_model_name, "multilingual" if "multilingual" in model_name else "english")
                return os.path.exists(model_path)
            else:
                return False
        return True

    def update_elapsed_time(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.elapsed_time_label.setText(f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    def update_gpu_status(self):
        gpu_info = "GPU Status: "
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_info += f"Available ({gpu_name})"
            self.gpu_label.setText(gpu_info)
            self.gpu_label.setStyleSheet("color: green;")
            
            # Prepare detailed GPU info
            detailed_info = f"CUDA Version: {torch.version.cuda}\n"
            detailed_info += f"cuDNN Version: {torch.backends.cudnn.version()}\n"
            detailed_info += f"Device Count: {torch.cuda.device_count()}\n"
            detailed_info += f"Current Device: {torch.cuda.current_device()}\n"
        else:
            gpu_info += "Not Available"
            self.gpu_label.setText(gpu_info)
            self.gpu_label.setStyleSheet("color: red;")
            
            # Prepare detailed GPU info for unavailable case
            detailed_info = "Possible reasons:\n"
            detailed_info += "1. CUDA is not installed\n"
            detailed_info += "2. PyTorch is not built with CUDA support\n"
            detailed_info += "3. NVIDIA drivers are not properly installed\n"
        
        self.detailed_gpu_info.setText(detailed_info)
        print(gpu_info)  # This will print the GPU info to the console for debugging

    def toggle_gpu_info(self, event):
        if self.detailed_gpu_info.isHidden():
            self.detailed_gpu_info.show()
        else:
            self.detailed_gpu_info.hide()

    def clear_cache(self):
        reply = QMessageBox.question(self, 'Clear Cache',
                                    "Are you sure you want to clear the cache? This will delete all downloaded models.",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            TranscriptionThread.clear_cache()
            QMessageBox.information(self, 'Cache Cleared', 'The cache has been successfully cleared.')

    def open_llm_settings(self):
        dialog = LLMSettingsDialog(self)
        if dialog.exec():
            self.llm_base_url = dialog.url_input.text().strip()
            self.llm_model = dialog.model_combo.currentText()
            self.summarize_button.setEnabled(bool(self.llm_base_url and self.llm_model))
            QMessageBox.information(self, "LLM Settings", "LLM settings updated successfully.")
        else:
            QMessageBox.information(self, "LLM Settings", "LLM settings update cancelled.")

    def summarize_transcript(self):
        if not self.llm_base_url or not self.llm_model:
            QMessageBox.warning(self, "Error", "Please set up LLM settings first.")
            return

        if not self.output_file_path or not os.path.exists(self.output_file_path):
            QMessageBox.warning(self, "Error", "No transcript file found. Please transcribe an audio file first.")
            return

        try:
            with open(self.output_file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()

            prompt = f"""You are an expert note taker and professional summarizer with a keen ability to distill complex information into clear, concise, and accurate summaries. Your task is to create a comprehensive yet concise summary of the following transcript from a meeting or presentation. Please adhere to these guidelines:

1. Identify and highlight the main topics, key points, and any crucial decisions or action items discussed.
2. Organize the summary in a logical structure, using bullet points or numbered lists where appropriate.
3. Preserve important details, statistics, or specific examples that substantiate main points.
4. If multiple speakers are involved, note any significant differences in opinions or perspectives.
5. Conclude with a brief overview of the most important takeaways or next steps, if applicable.
6. Aim for clarity and brevity while ensuring all vital information is captured.
7. Use professional language and maintain an objective tone.
8. Provide only the summary content without any introductory or concluding remarks.

Here's the transcript to summarize:

{transcript}

Please provide your expert summary:"""
            
            self.update_progress(0)
            self.text_edit.clear()
            self.text_edit.append("Starting summarization process...")
            
            response = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                summary = response.json()['response']
                
                # Clean up the summary
                prefixes_to_remove = [
                    "Here's a summary of the transcript:",
                    "Here is a summary of the transcript:",
                    "Based on the provided text, here is an expert summary:",
                    "Here's an expert summary of the transcript:",
                    "Summary of the transcript:",
                    "Here's a summary of the meeting:",
                    "Here's a concise summary of the conversation:",
                    "Here is a comprehensive summary of the meeting:",
                ]
                for prefix in prefixes_to_remove:
                    if summary.startswith(prefix):
                        summary = summary[len(prefix):].strip()
                
                suffixes_to_remove = [
                    "Let me know if you need any clarification or have any questions.",
                    "Let me know if there is anything I can clarify or expand upon.",
                    "Is there anything else you would like me to explain or elaborate on?",
                    "Please let me know if you need any additional information or clarification.",
                ]
                for suffix in suffixes_to_remove:
                    if summary.endswith(suffix):
                        summary = summary[:-len(suffix)].strip()
                
                summary_file = os.path.splitext(self.output_file_path)[0] + "_summary.txt"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                self.text_edit.clear()
                self.text_edit.append("Summary of Transcript:")
                self.text_edit.append(summary)
                
                self.update_progress(100)
                QMessageBox.information(self, "Summary Generated", f"Summary has been saved to {summary_file} and displayed in the main window.")
            else:
                self.text_edit.append(f"Error: Failed to generate summary. Status code: {response.status_code}")
                QMessageBox.warning(self, "Error", f"Failed to generate summary. Status code: {response.status_code}")
            
            self.update_progress(0)  # Reset progress bar

        except Exception as e:
            self.text_edit.append(f"Error: An error occurred while generating the summary: {str(e)}")
            QMessageBox.warning(self, "Error", f"An error occurred while generating the summary: {str(e)}")
            self.update_progress(0)  # Reset progress bar

    def quit_application(self):
        self.cancel_transcription()
        QApplication.quit()

class LLMSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LLM Settings")
        self.setGeometry(300, 300, 400, 200)
        
        layout = QVBoxLayout()
        
        # Ollama base URL input
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("Ollama Base URL:"))
        self.url_input = QLineEdit()
        self.url_input.setText("http://127.0.0.1:11434")
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.fetch_button = QPushButton("Fetch Models")
        self.fetch_button.clicked.connect(self.fetch_models)
        button_layout.addWidget(self.fetch_button)
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def fetch_models(self):
        base_url = self.url_input.text().strip()
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()['models']
                self.model_combo.clear()
                self.model_combo.addItems([model['name'] for model in models])
            else:
                QMessageBox.warning(self, "Error", f"Failed to fetch models. Status code: {response.status_code}")
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to connect to Ollama: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())