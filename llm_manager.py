import json
import os
import requests
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QComboBox, QPushButton, QTabWidget, QWidget, QMessageBox)
from openai import OpenAI
import anthropic

class LLMManager:
    def __init__(self):
        self.settings = self.load_settings()

    def load_settings(self):
        try:
            with open('llm_settings.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "current_provider": "local",
                "local": {"base_url": "http://127.0.0.1:11434", "model": "llama3:latest"},
                "openai": {"api_key": "", "model": "gpt-4o-mini-2024-07-18"},
                "anthropic": {"api_key": "", "model": "claude-2"}
            }

    def save_settings(self):
        with open('llm_settings.json', 'w') as f:
            json.dump(self.settings, f)

    def get_current_provider(self):
        return self.settings["current_provider"]

    def get_provider_settings(self, provider):
        return self.settings.get(provider, {})

    def open_settings_dialog(self, parent=None):
        dialog = LLMSettingsDialog(self.settings, parent)
        if dialog.exec():
            self.settings = dialog.settings
            self.save_settings()
            return True
        return False

    def summarize(self, text, summary_type):
        provider = self.get_current_provider()
        settings = self.get_provider_settings(provider)

        if provider == "local":
            return self.summarize_local(text, summary_type, settings)
        elif provider == "openai":
            return self.summarize_openai(text, summary_type, settings)
        elif provider == "anthropic":
            return self.summarize_anthropic(text, summary_type, settings)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def summarize_local(self, text, summary_type, settings):
        prompt = self.get_summary_prompt(summary_type, text)
        response = requests.post(
            f"{settings['base_url']}/api/generate",
            json={
                "model": settings["model"],
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Failed to generate summary. Status code: {response.status_code}")

    def summarize_openai(self, text, summary_type, settings):
        os.environ["OPENAI_API_KEY"] = settings["api_key"]
        client = OpenAI()
        prompt = self.get_summary_prompt(summary_type, text)
        response = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def summarize_anthropic(self, text, summary_type, settings):
        client = anthropic.Anthropic(api_key=settings["api_key"])
        prompt = self.get_summary_prompt(summary_type, text)
        response = client.completions.create(
            model=settings["model"],
            prompt=f"Human: {prompt}\n\nAssistant:",
            max_tokens_to_sample=1000
        )
        return response.completion

    def get_summary_prompt(self, summary_type, text):
        if summary_type == "General Summary":
            return f"Provide a concise summary of the following text:\n\n{text}"
        elif summary_type == "Meeting Minutes":
            return f"Create meeting minutes from the following text. Include key discussion points, decisions made, and action items:\n\n{text}"
        elif summary_type == "Action Items":
            return f"Extract and list all action items from the following text:\n\n{text}"
        else:
            raise ValueError(f"Unsupported summary type: {summary_type}")

class LLMSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("LLM Settings")
        self.setGeometry(300, 300, 400, 350)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("LLM Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Local", "OpenAI", "Anthropic"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        provider_layout.addWidget(self.provider_combo)
        layout.addLayout(provider_layout)

        # Settings tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_local_tab(), "Local")
        self.tabs.addTab(self.create_openai_tab(), "OpenAI")
        self.tabs.addTab(self.create_anthropic_tab(), "Anthropic")
        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Load settings
        self.load_settings()

    def create_local_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Ollama Base URL:"))
        self.local_url_input = QLineEdit()
        layout.addWidget(self.local_url_input)
        layout.addWidget(QLabel("Model:"))
        self.local_model_combo = QComboBox()
        self.local_model_combo.setEditable(True)
        layout.addWidget(self.local_model_combo)
        fetch_button = QPushButton("Fetch Models")
        fetch_button.clicked.connect(self.fetch_local_models)
        layout.addWidget(fetch_button)
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_openai_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("API Key:"))
        self.openai_key_input = QLineEdit()
        self.openai_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.openai_key_input)
        layout.addWidget(QLabel("Model:"))
        self.openai_model_combo = QComboBox()
        self.openai_model_combo.setEditable(True)
        available_models = [
            "gpt-4o-mini-2024-07-18",  # Default model
            "GPT-4o",
            "gpt-4o-2024-05-13",
            "GPT-4o mini",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-instruct"
        ]
        self.openai_model_combo.addItems(available_models)
        layout.addWidget(self.openai_model_combo)
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_anthropic_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("API Key:"))
        self.anthropic_key_input = QLineEdit()
        self.anthropic_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.anthropic_key_input)
        layout.addWidget(QLabel("Model:"))
        self.anthropic_model_input = QLineEdit()
        layout.addWidget(self.anthropic_model_input)
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def load_settings(self):
        # Load the current provider
        current_provider = self.settings.get("current_provider", "local").lower()
        print(f"Loading provider: {current_provider}")  # Debug print

        # Case-insensitive matching for provider
        for i in range(self.provider_combo.count()):
            if self.provider_combo.itemText(i).lower() == current_provider:
                self.provider_combo.setCurrentIndex(i)
                break
        else:
            print(f"Warning: Provider '{current_provider}' not found in combo box. Using default.")
            self.provider_combo.setCurrentIndex(0)  # Set to first item as default

        # Load settings for each provider
        local_settings = self.settings.get("local", {})
        self.local_url_input.setText(local_settings.get("base_url", ""))
        self.local_model_combo.setCurrentText(local_settings.get("model", ""))

        openai_settings = self.settings.get("openai", {})
        self.openai_key_input.setText(openai_settings.get("api_key", ""))
        saved_model = openai_settings.get("model", "")
        if saved_model:
            index = self.openai_model_combo.findText(saved_model)
            if index >= 0:
                self.openai_model_combo.setCurrentIndex(index)
            else:
                self.openai_model_combo.setCurrentText(saved_model)

        anthropic_settings = self.settings.get("anthropic", {})
        self.anthropic_key_input.setText(anthropic_settings.get("api_key", ""))
        self.anthropic_model_input.setText(anthropic_settings.get("model", ""))

        # Ensure the correct tab is displayed
        self.on_provider_changed(self.provider_combo.currentText())

    def save_settings(self):
        self.settings["current_provider"] = self.provider_combo.currentText().lower()
        
        self.settings["local"] = {
            "base_url": self.local_url_input.text(),
            "model": self.local_model_combo.currentText()
        }
        
        self.settings["openai"] = {
            "api_key": self.openai_key_input.text(),
            "model": self.openai_model_combo.currentText()
        }
        
        self.settings["anthropic"] = {
            "api_key": self.anthropic_key_input.text(),
            "model": self.anthropic_model_input.text()
        }

        print(f"Saving settings: {self.settings}")  # Debug print

        QMessageBox.information(self, "Settings Saved", "LLM settings have been saved successfully.")
        self.accept()

    def on_provider_changed(self, provider):
        index = ["Local", "OpenAI", "Anthropic"].index(provider)
        self.tabs.setCurrentIndex(index)

    def fetch_local_models(self):
        base_url = self.local_url_input.text().strip()
        if not base_url:
            QMessageBox.warning(self, "Error", "Please enter the Ollama Base URL.")
            return

        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                self.local_model_combo.clear()
                for model in models:
                    self.local_model_combo.addItem(model['name'])
                QMessageBox.information(self, "Success", f"Found {len(models)} models.")
            else:
                QMessageBox.warning(self, "Error", f"Failed to fetch models. Status code: {response.status_code}")
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to connect to Ollama: {str(e)}")

# You might want to add this at the end of the file for testing purposes
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    llm_manager = LLMManager()
    llm_manager.open_settings_dialog()
    sys.exit(app.exec())