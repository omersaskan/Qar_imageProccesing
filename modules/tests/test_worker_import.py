import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

class TestWorkerImport(unittest.TestCase):
    def test_import_worker(self):
        """
        Verify that IngestionWorker can be imported without instant crash.
        This catches global-scope initialization errors like the OutputManifest bug.
        """
        try:
            from modules.operations.worker import IngestionWorker
            # Also check the singleton instance if it exists
            from modules.operations.worker import worker_instance
            
            self.assertIsNotNone(IngestionWorker)
            self.assertIsNotNone(worker_instance)
            print("Successfully imported IngestionWorker and singleton instance.")
        except Exception as e:
            self.fail(f"Worker import failed: {e}")

if __name__ == "__main__":
    unittest.main()
