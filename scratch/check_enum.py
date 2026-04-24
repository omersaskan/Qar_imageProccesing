
from enum import Enum
class E(str, Enum):
    A = "a"
print(f"E.A == 'a': {E.A == 'a'}")
print(f"str(E.A) == 'a': {str(E.A) == 'a'}")
print(f"E.A.value == 'a': {E.A.value == 'a'}")
