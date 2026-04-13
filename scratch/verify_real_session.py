from pathlib import Path
from modules.reconstruction_engine.adapter import COLMAPAdapter

def verify_gate():
    adapter = COLMAPAdapter()
    # Use the existing successful sparse model from the failed session
    sparse_dir = Path(r"c:\modelPlate\data\reconstructions\job_cap_24b4136c\attempt_1_denser_frames\sparse")
    
    print(f"Checking existing sparse workspace: {sparse_dir}")
    
    class MockLog:
        def write(self, msg):
            print(f"LOG: {msg}")

    # This calls _parse_model_stats internally for each model (0, 1, 2...)
    best_model = adapter._select_best_sparse_model(sparse_dir, MockLog())
    
    if best_model:
        num_images = best_model["registered_images"]
        num_points = best_model["points_3d"]
        print(f"\nBest Model Selected: {best_model['path'].name}")
        print(f"Images: {num_images}")
        print(f"Points: {num_points}")
        
        # This is the exact check in adapter.py:run_reconstruction
        if num_images >= 5 and num_points >= 100:
            print("\nSUCCESS: Sparse gate CLEARED.")
            print("The pipeline would proceed to 'image_undistorter' and dense stages.")
        else:
            print(f"\nFAILURE: Gate still blocked. (images={num_images}, points={num_points})")
    else:
        print("\nFAILURE: No model found at all.")

if __name__ == "__main__":
    verify_gate()
