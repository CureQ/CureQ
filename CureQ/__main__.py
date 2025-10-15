# pyinstaller -n "MEAlytics" __main__.py --icon="MEAlytics_logo.ico" --add-data="./GUI/MEAlytics_logo.ico":"." --add-data="./GUI/theme.json":"." -y

import argparse
import os
from importlib.metadata import version
import multiprocessing

from CureQ.GUI.mea_analysis_tool import MEA_GUI

def launch_gui():
    """GUI launch function"""
    MEA_GUI()

def create_shortcut():
    try:
        script_path = str(os.path.abspath(__file__))
        from pyshortcuts import make_shortcut
        
        make_shortcut(script=script_path, 
                      name="MEAlytics",
                      icon=os.path.join(os.path.dirname(__file__), "MEAlytics_logo.ico"),
                      desktop=True,
                      startmenu=True)
        
        print("Succesfully created desktop shortcut")
    except Exception as error:
        print(f"Failed to create shortcut:\n{error}")

def print_version():
    print(f"MEAlytics - Version: {version('CureQ')}")

def main():
    parser = argparse.ArgumentParser(description='Launch CureQ GUI')
    parser.add_argument('--create-shortcut', action='store_true', help='Create a desktop shortcut')
    parser.add_argument('--version', action='store_true', help='Add shortcut to Start Menu')
    args = parser.parse_args()
    
    if args.create_shortcut:
        create_shortcut()
    elif args.version:
        print_version()
    else:
        launch_gui()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as error:
        print(error)