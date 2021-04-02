import os

def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))

def get_output_dir():
    out = os.path.join(get_project_dir(), 'out')
    os.makedirs(out, exist_ok=True)
    return out

def get_data_dir():
    return os.path.join(get_project_dir(), 'data')
