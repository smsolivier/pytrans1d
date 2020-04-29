import warnings 
from pathlib import Path 

def warning_format(message, category, filename, lineno, line=None):
	path = Path(*Path(filename).parts[-3:])
	return '{}: {}:{} {}\n'.format(category.__name__, str(path), lineno, message)
warnings.formatwarning = warning_format

class ToleranceWarning(UserWarning):
	pass 

class NegativityWarning(UserWarning):
	pass 
