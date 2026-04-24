import sys
import os
from pathlib import Path

# Add modules to path
modules_path = str(Path(__file__).parent.parent.parent)
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

import numpy as np
import cv2
from capture_workflow.quality_analyzer import QualityAnalyzer

def test_direct():
    analyzer = QualityAnalyzer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    analysis = analyzer.analyze_frame(frame)
    print(f"Analysis Result: {analysis}")

if __name__ == "__main__":
    try:
        test_direct()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
