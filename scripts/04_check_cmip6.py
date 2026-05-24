"""Check CMIP6 data source API"""
from earth2studio.data import CMIP6
import inspect
print("CMIP6 signature:")
print(inspect.signature(CMIP6.__init__))
print()
print("CMIP6 call signature:")
print(inspect.signature(CMIP6.__call__))
print()
# Check docstring
print("CMIP6 docstring:")
print(CMIP6.__doc__[:2000] if CMIP6.__doc__ else "No docstring")