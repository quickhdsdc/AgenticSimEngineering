import importlib.util, os, sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './IM')))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GENCODE_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..", "IM", "gencodeforAMG")
)

print("GENCODE PATH →", GENCODE_PATH)

def load_func_from_file(file_name, delay_folder=GENCODE_PATH):

    
    """Dynamically loads a Python file from a given folder and retrieves the processing delay function."""
    file_path = os.path.join(delay_folder, file_name)
    if os.path.exists(file_path):
        module_name = file_name.replace(".py", "")  # Remove the .py extension
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Assume the file has a function named `generate_data()` that returns the processing delay
        delay_func = getattr(module, "generate_data", None)
        if delay_func and callable(delay_func):
            return delay_func()
        else:
            raise AttributeError(f"Function 'generate_data' not found in {file_name}.")
    else:
        raise FileNotFoundError(f"File {file_name} not found in folder {delay_folder}.")


#load_func_from_file("M1_processing_delay_code.py")