import sys

try:
    import pytesseract
    from PIL import Image
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you are running this in your virtual environment.")
    sys.exit(1)

def test_tesseract():
    print("1. Checking if pytesseract can find tesseract in PATH...")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"   [SUCCESS] Found tesseract version: {version}")
        return True
    except Exception as e:
        print(f"   [FAILED] Could not find tesseract: {e}")
        
    print("\n2. Trying to explicitly set tesseract_cmd to 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'...")
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    try:
        version = pytesseract.get_tesseract_version()
        print(f"   [SUCCESS] Found tesseract version: {version}")
        print("\n=== CONCLUSION ===")
        print("Tesseract is installed in C:\\Program Files\\Tesseract-OCR\\tesseract.exe but is NOT recognized in the current terminal's PATH.")
        print("Options to fix this:")
        print("  Option A: Close all terminal windows, restart VS Code/your terminal so it loads the new PATH variable.")
        print("  Option B: Add `import pytesseract; pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'` at the top of your extractor code.")
        return True
    except Exception as e:
        print(f"   [FAILED] Still could not find tesseract: {e}")
        print("Tesseract does not seem to be installed at the default location.")
        return False

if __name__ == "__main__":
    test_tesseract()
