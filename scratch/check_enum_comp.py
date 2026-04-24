
from enum import Enum
class E(str, Enum):
    A = "a"
print(f"E.A == 'a': {E.A == 'a'}")
print(f"'a' == E.A: {'a' == E.A}")
