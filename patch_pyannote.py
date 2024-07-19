import os
import sys
from pathlib import Path
from config import CACHE_DIR
from huggingface_hub import constants

def patch_huggingface_hub():
    # Set the cache directory for huggingface_hub
    constants.HUGGINGFACE_HUB_CACHE = os.path.join(CACHE_DIR, 'huggingface')
    
    # Ensure the directory exists
    os.makedirs(constants.HUGGINGFACE_HUB_CACHE, exist_ok=True)
    
    print(f"Hugging Face Hub cache set to: {constants.HUGGINGFACE_HUB_CACHE}")

def patch_os_path():
    original_join = os.path.join
    original_expanduser = os.path.expanduser

    def patched_join(*args):
        path = original_join(*args)
        if '.cache' in str(path):
            return str(path).replace(str(original_expanduser('~/.cache')), CACHE_DIR)
        return path

    def patched_expanduser(path):
        path_str = str(path)
        if path_str.startswith('~/.cache'):
            return Path(path_str.replace('~/.cache', CACHE_DIR))
        return original_expanduser(path)

    os.path.join = patched_join
    os.path.expanduser = patched_expanduser

# Apply the patches
patch_huggingface_hub()
patch_os_path()

print("Patches applied successfully.")