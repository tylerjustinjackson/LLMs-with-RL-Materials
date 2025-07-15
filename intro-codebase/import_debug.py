import subprocess, sys


def install_package(package):
    """
    Installs a Python package using pip if it's not already installed.
    """
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            sys.exit(1)
