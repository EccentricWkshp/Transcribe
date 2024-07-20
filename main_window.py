import ffmpeg
import os
import sys
import time
import re
import json
import requests

from PyQt6.QtWidgets import (QMainWindow, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, QHBoxLayout,
                             QWidget, QProgressBar, QLabel, QComboBox, QMessageBox, QMenuBar, QMenu,
                             QCheckBox, QSizePolicy, QApplication, QDialog, QLineEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup, QInputDialog,
                             QTabWidget)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
import torch

from llm_manager import LLMSettingsDialog, LLMManager
from model_manager import ModelManagerDialog, ModelUtils
from summarization_thread import SummarizationThread
from transcription_thread import TranscriptionThread

from config import WHISPER_MODELS, MODELS_DIR, PYANNOTE_AUTH_TOKEN

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
        self.llm_manager = LLMManager()
        self.llm_base_url = None
        self.llm_model = None
        self.load_llm_settings()  # Load saved LLM settings

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

        load_transcript_action = QAction('Load Transcript', self)
        load_transcript_action.triggered.connect(self.load_transcript)
        file_menu.addAction(load_transcript_action)

        load_diarization_action = QAction('Load Diarization', self)
        load_diarization_action.triggered.connect(self.load_diarization)
        file_menu.addAction(load_diarization_action)

        export_submenu = file_menu.addMenu('Export')
        export_txt_action = QAction('Export as TXT', self)
        export_txt_action.triggered.connect(lambda: self.export_transcript('txt'))
        export_submenu.addAction(export_txt_action)
        export_srt_action = QAction('Export as SRT', self)
        export_srt_action.triggered.connect(lambda: self.export_transcript('srt'))
        export_submenu.addAction(export_srt_action)
        export_vtt_action = QAction('Export as VTT', self)
        export_vtt_action.triggered.connect(lambda: self.export_transcript('vtt'))
        export_submenu.addAction(export_vtt_action)

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

        self.label_speakers_button = QPushButton("Label Speakers")
        self.label_speakers_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.label_speakers_button.clicked.connect(self.open_speaker_naming_dialog)
        self.label_speakers_button.setEnabled(False)
        button_layout.addWidget(self.label_speakers_button)

        # Clear button
        self.clear_button = QPushButton("Clear All Tabs")
        self.clear_button.clicked.connect(self.clear_all_tabs)
        button_layout.addWidget(self.clear_button)

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

        self.tabbed_output = QTabWidget()
        self.tabbed_output.setTabsClosable(True)
        self.tabbed_output.tabCloseRequested.connect(self.close_tab)
        main_layout.addWidget(self.tabbed_output)

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

    def clear_all_tabs(self):
        reply = QMessageBox.question(self, 'Clear All Tabs',
                                     "Are you sure you want to clear all tabs?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.tabbed_output.clear()

    def close_tab(self, index):
        self.tabbed_output.removeTab(index)

    def add_new_tab(self, content, title):
        new_tab = QTextEdit()
        new_tab.setReadOnly(True)
        new_tab.setPlainText(content)
        self.tabbed_output.addTab(new_tab, title)
        self.tabbed_output.setCurrentWidget(new_tab)
        return new_tab

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
        file_formats = "Audio/Video Files (*.mp3 *.mp4 *.wav *.avi *.mov *.flac *.ogg *.m4a *.webm)"
        self.input_file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio/Video File", "", file_formats)
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
                probe = ffmpeg.probe(self.input_file_path)
                duration = float(probe['streams'][0]['duration'])
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                self.file_length_label.setText(f"File Length: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            except ffmpeg.Error as e:
                self.file_length_label.setText(f"File Length: Unable to determine")
                print(f"Error determining file length: {e.stderr.decode()}")
            except Exception as e:
                self.file_length_label.setText(f"File Length: Unable to determine")
                print(f"Unexpected error determining file length: {str(e)}")
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

        self.add_new_tab("", f"Transcription {self.tabbed_output.count() + 1}")
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
        current_tab = self.tabbed_output.currentWidget()
        if isinstance(current_tab, QTextEdit):
            for label, name in self.speaker_names.items():
                text = text.replace(f"[{label}]:", f"[{name}]:")
            current_tab.append(text)

    def handle_speaker_labels(self, speaker_labels):
        self.speaker_naming_dialog = SpeakerNamingDialog(speaker_labels, self)
        self.speaker_naming_dialog.namesUpdated.connect(self.update_speaker_names)
        self.speaker_naming_dialog.show()

    def update_speaker_names(self, names):
        self.speaker_names = names
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.assign_speaker_names(self.speaker_names)
        self.update_transcript_file()
        self.update_transcript_display()

    def update_transcript_display(self):
        if self.output_file_path:
            with open(self.output_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.add_new_tab(content, f"Transcript {self.tabbed_output.count() + 1}")
    
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

    def finalize_transcription(self):
        self.reset_ui()
        end_time = time.time()
        self.end_time_label.setText(f"End Time: {time.strftime('%H:%M:%S')}")
        total_time = end_time - self.start_time
        self.elapsed_time_label.setText(f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        self.timer.stop()
        self.update_transcript_display()
        QMessageBox.information(self, "Transcription Complete", "The transcription has been saved successfully.")
        self.label_speakers_button.setEnabled(True)
        self.summarize_button.setEnabled(True)
        self.open_speaker_naming_dialog()  # Automatically open the speaker naming dialog

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

    def load_llm_settings(self):
        try:
            with open('llm_settings.json', 'r') as f:
                settings = json.load(f)
                self.llm_base_url = settings.get('base_url')
                self.llm_model = settings.get('model')
        except FileNotFoundError:
            pass  # No saved settings yet

    def save_llm_settings(self):
        settings = {
            'base_url': self.llm_base_url,
            'model': self.llm_model
        }
        with open('llm_settings.json', 'w') as f:
            json.dump(settings, f)

    def open_llm_settings(self):
        if self.llm_manager.open_settings_dialog(self):
            # Settings were saved, you might want to update UI or reload some components
            pass

    def load_transcript(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Transcript File", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content:
                self.add_new_tab(content, f"Loaded Transcript {self.tabbed_output.count() + 1}")
                self.output_file_path = file_path
                self.label_speakers_button.setEnabled(True)
                self.summarize_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Invalid Transcript", "The selected file does not contain a valid transcript.")
    
    def load_diarization(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Diarization File", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if self.is_valid_transcript(content):
                self.add_new_tab(content, f"Loaded Diarization {self.tabbed_output.count() + 1}")
                self.output_file_path = file_path
                self.label_speakers_button.setEnabled(True)
                self.summarize_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Invalid Diarization", "The selected file does not contain a valid diarization with speaker labels.")

    def is_valid_transcript(self, content):
        # Check if the transcript contains speaker labels in various formats
        speaker_pattern = r'\[((?:SPEAKER_\d+|[A-Za-z\s]+))\]:'
        return bool(re.search(speaker_pattern, content))

    def open_speaker_naming_dialog(self):
        if not self.output_file_path:
            QMessageBox.warning(self, "No Transcript", "Please load or create a transcript first.")
            return

        with open(self.output_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        speaker_pattern = r'\[((?:SPEAKER_\d+|[A-Za-z\s]+))\]:'
        speaker_labels = list(set(re.findall(speaker_pattern, content)))
        
        if not speaker_labels:
            QMessageBox.warning(self, "No Speakers Found", "No speaker labels found in the transcript.")
            return

        self.speaker_naming_dialog = SpeakerNamingDialog(speaker_labels, self)
        self.speaker_naming_dialog.namesUpdated.connect(self.update_speaker_names)
        self.speaker_naming_dialog.show()

    def summarize_transcript(self):
        if not self.output_file_path or not os.path.exists(self.output_file_path):
            QMessageBox.warning(self, "Error", "No transcript file found. Please transcribe an audio file first.")
            return

        summary_type, ok = QInputDialog.getItem(self, "Summary Type", 
                                                "Select the type of summary:", 
                                                ["General Summary", "Meeting Minutes", "Action Items"], 
                                                0, False)
        if not ok:
            return

        try:
            with open(self.output_file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()

            self.start_time = time.time()
            self.start_time_label.setText(f"Start Time: {time.strftime('%H:%M:%S')}")
            self.end_time_label.setText("End Time: Not finished")
            self.elapsed_time_label.setText("Elapsed Time: 00:00:00")
            self.timer.start(1000)  # Update every second

            self.summarization_thread = SummarizationThread(self.llm_manager, transcript, summary_type)
            self.summarization_thread.progress.connect(self.update_progress)
            self.summarization_thread.summary_chunk.connect(self.append_summary)
            self.summarization_thread.summary_complete.connect(self.finalize_summary)
            self.summarization_thread.error_occurred.connect(self.handle_summarization_error)
            self.summarization_thread.start()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred while generating the summary: {str(e)}")
            self.update_progress(0)  # Reset progress bar

    def append_summary(self, text):
        current_tab = self.tabbed_output.currentWidget()
        if isinstance(current_tab, QTextEdit):
            current_tab.append(text)
    
    def finalize_summary(self, final_summary):
        end_time = time.time()
        self.end_time_label.setText(f"End Time: {time.strftime('%H:%M:%S')}")
        total_time = end_time - self.start_time
        self.elapsed_time_label.setText(f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        self.timer.stop()

        final_summary = self.clean_summary(final_summary)
        summary_type = self.summarization_thread.summary_type
        summary_file = os.path.splitext(self.output_file_path)[0] + f"_{summary_type.lower().replace(' ', '_')}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(final_summary)

        self.add_new_tab(f"{summary_type}:\n\n{final_summary}", summary_type)
        QMessageBox.information(self, "Summary Generated", f"{summary_type} has been saved to {summary_file} and displayed in a new tab.")
    
    def handle_summarization_error(self, error_message):
        self.timer.stop()
        QMessageBox.warning(self, "Error", f"An error occurred while generating the summary: {error_message}")
        self.update_progress(0)  # Reset progress bar

    def get_summary_prompt(self, summary_type, chunk):
        if summary_type == "General Summary":
            return f"""Summarize the following chunk of conversation transcript. Focus on key points, decisions, and important information:

{chunk}

Provide a concise summary:"""
        elif summary_type == "Meeting Minutes":
            return f"""Please create detailed meeting minutes from the following transcript. Include:

1. Date, time, and all attendees mentioned
2. All agenda items discussed
3. Key points of discussion for each agenda item, including:
    a. Detailed summaries of reports or updates given
    b. Important questions raised and answers provided
    c. Significant opinions or concerns expressed by participants
4. Any decisions made or actions agreed upon
5. Specific assignments or action items for individuals or groups
6. Any voting results
7. Upcoming events, future meeting dates, or deadlines mentioned
8. Brief summaries of any presentations or guest speakers
9. Notable quotes or important statements from participants

Organize the information clearly under appropriate headings, maintaining the original sequence of topics as discussed in the meeting. Include relevant details but summarize lengthy discussions. Use professional language and format the minutes in a clear, easy-to-read structure.
If exact dates, times, or names are unclear, use placeholders and add a note to fill in the correct information before finalizing the minutes.
Please pay special attention to capturing the essence of program updates, committee reports, and any detailed discussions on specific projects or initiatives.
Anyone not attending the meeting should be able to read the minutes and get an understanding of the discussion.

Here's the transcript to summarize:

Chunk:
{chunk}

Provide structured meeting minutes:"""
        elif summary_type == "Action Items":
            return f"""Extract all action items from the following chunk of conversation transcript. An action item should include:
1. The task to be done
2. The person responsible (if mentioned)
3. The deadline (if mentioned)

Chunk:
{chunk}

List all action items:"""

    def get_final_summary_prompt(self, summary_type, combined_summary):
        if summary_type == "General Summary":
            return f"""Create a comprehensive summary of the following transcript summaries. Ensure all major topics and key points are included:

{combined_summary}

Provide a final, coherent summary:"""
        elif summary_type == "Meeting Minutes":
            return f"""Create a final, comprehensive set of meeting minutes from the following summaries. Your task is to:

1. Compile all information into a single, coherent document
2. Maintain the chronological order of discussions
3. Eliminate any redundancies
4. Ensure a professional, easy-to-read format with clear headings and subheadings
5. Include all key points, discussions, decisions, action items, and assignments
6. Standardize the language and style throughout the document

The minutes should include:
- Meeting date, time, and attendees (if available)
- Agenda items discussed
- Key points of discussion for each item with a summary of the discussion
- Decisions made and actions agreed upon
- Important updates or reports
- Assignments or action items for individuals or groups
- Voting results (if any)
- Upcoming events or future meeting dates
- Summaries of any presentations or guest speakers

Be sure to organize the information clearly under appropriate headings. Maintain the original sequence of topics as they were discussed in the meeting.
Include relevant details but summarize lengthy discussions. Use professional language and format the minutes in a clear, easy-to-read structure.
If exact dates, times, or names are unclear, use placeholders and add a note to fill in the correct information before finalizing the minutes.

Here are the summaries to compile:
{combined_summary}

Provide the final meeting minutes:"""
        elif summary_type == "Action Items":
            return f"""Compile and organize the following list of action items. Remove any duplicates and group related items. For each action item, include:
1. The task to be done
2. The person responsible (if mentioned)
3. The deadline (if mentioned)

Action items:
{combined_summary}

Provide a final, organized list of action items:"""

    def clean_summary(self, summary):
        prefixes_to_remove = [
            "Here's a summary of the transcript:",
            "Here is a summary of the transcript:",
            "Based on the provided text, here is an expert summary:",
            "Here's an expert summary of the transcript:",
            "Summary of the transcript:",
            "Here's a summary of the meeting:",
            "Here's a concise summary of the conversation:",
            "Here is a comprehensive summary of the meeting:",
            "Meeting Minutes:",
        ]
        for prefix in prefixes_to_remove:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()

        suffixes_to_remove = [
            "Let me know if you need any clarification or have any questions.",
            "Let me know if there is anything I can clarify or expand upon.",
            "Is there anything else you would like me to explain or elaborate on?",
            "Please let me know if you need any additional information or clarification.",
            "*Please fill in any placeholders with relevant information.*",
        ]
        for suffix in suffixes_to_remove:
            if summary.endswith(suffix):
                summary = summary[:-len(suffix)].strip()

        return summary

    def export_transcript(self, format):
        if not self.output_file_path:
            QMessageBox.warning(self, "No Transcript", "Please transcribe or load a transcript first.")
            return

        if format == 'txt':
            export_path, _ = QFileDialog.getSaveFileName(self, "Export as TXT", "", "Text Files (*.txt)")
            if export_path:
                shutil.copy(self.output_file_path, export_path)
        elif format in ['srt', 'vtt']:
            export_path, _ = QFileDialog.getSaveFileName(self, f"Export as {format.upper()}", "", f"{format.upper()} Files (*.{format})")
            if export_path:
                self.convert_to_subtitle_format(export_path, format)

        if export_path:
            QMessageBox.information(self, "Export Complete", f"Transcript exported as {format.upper()} to {export_path}")

    def convert_to_subtitle_format(self, export_path, format):
        with open(self.output_file_path, 'r', encoding='utf-8') as f:
            transcript = f.read()

        # Split the transcript into segments (you may need to adjust this based on your transcript format)
        segments = re.split(r'\n(?=\[)', transcript)

        with open(export_path, 'w', encoding='utf-8') as f:
            if format == 'vtt':
                f.write("WEBVTT\n\n")

            for i, segment in enumerate(segments):
                start_time = i * 5  # Assume each segment is 5 seconds long (adjust as needed)
                end_time = start_time + 5

                if format == 'srt':
                    f.write(f"{i+1}\n")
                    f.write(f"{self.format_time(start_time)} --> {self.format_time(end_time)}\n")
                    f.write(f"{segment.strip()}\n\n")
                elif format == 'vtt':
                    f.write(f"{self.format_time(start_time, vtt=True)} --> {self.format_time(end_time, vtt=True)}\n")
                    f.write(f"{segment.strip()}\n\n")

    def format_time(self, seconds, vtt=False):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if vtt:
            return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
        else:
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d},000"

    def quit_application(self):
        self.cancel_transcription()
        QApplication.quit()

class SpeakerNamingDialog(QWidget):
    namesUpdated = pyqtSignal(dict)

    def __init__(self, speaker_labels, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())