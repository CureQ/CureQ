import argparse
import os
import sys
from pathlib import Path
from importlib.metadata import version

try:
    from .mea_analysis_tool import MEA_GUI
except:
    from mea_analysis_tool import MEA_GUI

def launch_gui():
    """GUI launch function"""
    MEA_GUI()

def add_to_start_menu():
    """Adds a shortcut to the Start Menu."""
    if sys.platform == "win32":
        try:
            import winshell
            from win32com.client import Dispatch
            
            start_menu = Path(winshell.programs()) / "CureQ"
            start_menu.mkdir(parents=True, exist_ok=True)
            
            shortcut_path = start_menu / "MEA Analysis Tool.lnk"
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = "-m CureQ.main"
            shortcut.IconLocation = os.path.join(os.path.dirname(__file__), "cureq_icon.ico")
            shortcut.save()
            
            print(f"Start Menu shortcut created at {shortcut_path}")
        except Exception as e:
            print(f"Failed to create Start Menu shortcut: {e}")
            input("Press Enter to exit...")

def create_shortcut():
    if sys.platform == "win32":
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = Path(winshell.desktop())
            shortcut_path = desktop / "MEA Analysis Tool.lnk"
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = "-m CureQ.main"
            shortcut.IconLocation = os.path.join(os.path.dirname(__file__), "cureq_icon.ico")
            shortcut.save()    
            print(f"Desktop shortcut created at {shortcut_path}")
        except Exception as e:
            print(f"Failed to create desktop shortcut: {e}")
            input("Press Enter to exit...")

def print_version():
    print(f"CureQ MEA analysis tool - Version: {version('CureQ')}")

def main():
    parser = argparse.ArgumentParser(description='Launch CureQ GUI')
    parser.add_argument('--create-shortcut', action='store_true', help='Create a desktop shortcut')
    parser.add_argument('--add-to-start-menu', action='store_true', help='Add shortcut to Start Menu')
    parser.add_argument('--version', action='store_true', help='Add shortcut to Start Menu')
    args = parser.parse_args()
    
    if args.create_shortcut:
        create_shortcut()
    elif args.add_to_start_menu:
        add_to_start_menu()
    elif args.version:
        print_version()
    else:
        launch_gui()

if __name__ == '__main__':
    main()