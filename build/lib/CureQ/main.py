import argparse
from mea_analysis_tool import MEA_GUI

def launch_gui():
    """GUI launch function"""
    MEA_GUI()
    

def main():
    parser = argparse.ArgumentParser(description='Launch the GUI application')
    args = parser.parse_args()
    
    launch_gui(args.param1, args.param2)

if __name__ == '__main__':
    main()