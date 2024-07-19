import os
import time
import requests
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, 
                             QRadioButton, QButtonGroup, QMessageBox, QProgressBar, 
                             QLabel, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from config import WHISPER_MODELS, MODELS_DIR

class DownloadSignals(QObject):
    progress = pyqtSignal(int, float)
    finished = pyqtSignal()
    error = pyqtSignal(str)

class DownloadThread(QThread):
    def __init__(self, url, file_path):
        super().__init__()
        self.url = url
        self.file_path = file_path
        self.signals = DownloadSignals()
        self.is_cancelled = False

    def run(self):
        try:
            print(f"DownloadThread: Starting download from {self.url}")
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024
            downloaded_size = 0
            start_time = time.time()

            with open(self.file_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    if self.is_cancelled:
                        print("DownloadThread: Download cancelled")
                        return
                    size = file.write(data)
                    downloaded_size += size
                    if total_size:
                        progress = int((downloaded_size / total_size) * 100)
                        elapsed_time = time.time() - start_time
                        speed = downloaded_size / (1024 * 1024 * elapsed_time)
                        self.signals.progress.emit(progress, speed)

            print("DownloadThread: Download completed successfully")
            self.signals.finished.emit()
        except Exception as e:
            print(f"DownloadThread: Error occurred - {str(e)}")
            self.signals.error.emit(str(e))

    def cancel(self):
        self.is_cancelled = True

class DownloadDialog(QDialog):
    def __init__(self, model_name, language, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Downloading {model_name} ({language}) Model")
        self.setFixedSize(400, 150)
        
        layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        self.speed_label = QLabel("Download speed: N/A")
        layout.addWidget(self.speed_label)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_download)
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
        
        self.download_cancelled = False

    def update_progress(self, progress, speed):
        self.progress_bar.setValue(progress)
        self.speed_label.setText(f"Download speed: {speed:.2f} MB/s")

    def cancel_download(self):
        self.download_cancelled = True
        self.reject()

class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Whisper Model Manager")
        self.setGeometry(200, 200, 500, 400)

        layout = QVBoxLayout()

        self.model_list = QListWidget()
        layout.addWidget(self.model_list)

        self.language_group = QButtonGroup(self)
        self.english_radio = QRadioButton("English")
        self.multilingual_radio = QRadioButton("Multilingual")
        self.english_radio.setChecked(True)
        self.language_group.addButton(self.english_radio)
        self.language_group.addButton(self.multilingual_radio)

        lang_layout = QHBoxLayout()
        lang_layout.addWidget(self.english_radio)
        lang_layout.addWidget(self.multilingual_radio)
        layout.addLayout(lang_layout)

        button_layout = QHBoxLayout()
        self.install_button = QPushButton("Install")
        self.install_button.clicked.connect(self.install_model)
        button_layout.addWidget(self.install_button)

        self.uninstall_button = QPushButton("Uninstall")
        self.uninstall_button.clicked.connect(self.uninstall_model)
        button_layout.addWidget(self.uninstall_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.english_radio.toggled.connect(self.update_model_list)
        self.multilingual_radio.toggled.connect(self.update_model_list)

        self.update_model_list()

    def update_model_list(self):
        print("Updating model list")
        self.model_list.clear()
        language = "english" if self.english_radio.isChecked() else "multilingual"
        for model, lang_data in WHISPER_MODELS.items():
            if language in lang_data:
                status = 'Installed' if self.is_model_installed(model, language) else 'Not Installed'
                size = lang_data[language]["size"]
                item = f"{model} ({language}) - {size} - {status}"
                self.model_list.addItem(item)
        print("Model list updated")

    def is_model_installed(self, model_name, language):
        model_filename = f"{model_name}.pt"
        model_path = os.path.join(MODELS_DIR, model_filename)
        exists = os.path.exists(model_path)
        print(f"Checking if model {model_name} ({language}) is installed: {exists}")
        return exists

    def install_model(self):
        print("Install model button clicked")
        selected_item = self.model_list.currentItem()
        if selected_item:
            model_info = selected_item.text().split(" - ")
            model_name = model_info[0].split(" (")[0]
            language = "english" if self.english_radio.isChecked() else "multilingual"
            if not self.is_model_installed(model_name, language):
                reply = QMessageBox.question(self, "Install Model", 
                                             f"Are you sure you want to install the {model_name} ({language}) model?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    print(f"Starting download for model {model_name} ({language})")
                    self.download_model(model_name, language)
            else:
                QMessageBox.information(self, "Model Already Installed", 
                                        f"The {model_name} ({language}) model is already installed.")
        self.update_model_list()

    def uninstall_model(self):
        print("Uninstall model button clicked")
        selected_item = self.model_list.currentItem()
        if selected_item:
            model_info = selected_item.text().split(" - ")
            model_name = model_info[0].split(" (")[0]
            language = "english" if self.english_radio.isChecked() else "multilingual"
            if self.is_model_installed(model_name, language):
                reply = QMessageBox.question(self, "Uninstall Model", 
                                             f"Are you sure you want to uninstall the {model_name} ({language}) model?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    print(f"Uninstalling model {model_name} ({language})")
                    self.remove_model(model_name, language)
            else:
                QMessageBox.information(self, "Model Not Installed", 
                                        f"The {model_name} ({language}) model is not installed.")
        self.update_model_list()

    def download_model(self, model_name, language):
        url = WHISPER_MODELS[model_name][language]["url"]
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_filename = f"{model_name}.pt"
        model_path = os.path.join(MODELS_DIR, model_filename)

        download_dialog = DownloadDialog(model_name, language, self)
        download_thread = DownloadThread(url, model_path)
        
        download_thread.signals.progress.connect(download_dialog.update_progress)
        download_thread.signals.finished.connect(download_dialog.accept)
        download_thread.signals.error.connect(download_dialog.reject)

        download_thread.start()
        result = download_dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            print(f"Download finished for model {model_name} ({language})")
            QMessageBox.information(self, "Download Complete", f"The {model_name} ({language}) model has been installed successfully.")
        else:
            download_thread.cancel()
            download_thread.wait()
            if download_dialog.download_cancelled:
                print(f"Download cancelled for model {model_name} ({language})")
                QMessageBox.information(self, "Download Cancelled", "The model download was cancelled.")
            else:
                print(f"Download error for model {model_name} ({language})")
                QMessageBox.critical(self, "Download Error", f"An error occurred while downloading the model. Please try again.")
            
            self.safe_remove_file(model_path)

        self.update_model_list()

    def safe_remove_file(self, file_path, max_attempts=5, delay=1):
        print(f"Attempting to remove file: {file_path}")
        for attempt in range(max_attempts):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"File removed successfully: {file_path}")
                return True
            except PermissionError:
                print(f"Permission error when trying to remove file (attempt {attempt + 1})")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                else:
                    print(f"Failed to remove file after {max_attempts} attempts")
                    QMessageBox.warning(self, "File Removal Failed", 
                                        f"Unable to remove the file: {file_path}. Please delete it manually.")
                    return False

    def remove_model(self, model_name, language):
        print(f"Removing model {model_name} ({language})")
        model_filename = f"{model_name}.pt"
        model_path = os.path.join(MODELS_DIR, model_filename)
        if self.safe_remove_file(model_path):
            print(f"Model {model_name} ({language}) uninstalled successfully")
            QMessageBox.information(self, "Uninstall Complete", f"The {model_name} ({language}) model has been uninstalled successfully.")
        else:
            print(f"Failed to uninstall model {model_name} ({language})")
            QMessageBox.critical(self, "Uninstall Error", f"An error occurred while uninstalling the model. Please try to delete the file manually: {model_path}")
        self.update_model_list()

class ModelUtils:
    @staticmethod
    def is_model_installed(model_name, language):
        model_filename = f"{model_name}.pt"
        model_path = os.path.join(MODELS_DIR, model_filename)
        return os.path.exists(model_path)