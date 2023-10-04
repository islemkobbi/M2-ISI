import os
import shutil


def remove_chache_folders(current_repo: str = os.path.dirname(__file__)):
    """
    removes all __pycache__ directories in the module

    Args:
        current_repo: current folder to clean recursively
    """
    new_refs = [current_repo + "/" + elem for elem in os.listdir(current_repo)]
    for elem in new_refs:
        if os.path.isdir(elem):
            if "__pycache__" in elem:
                shutil.rmtree(elem)
            else:
                remove_chache_folders(current_repo=elem)
