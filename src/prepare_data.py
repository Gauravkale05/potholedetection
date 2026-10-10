import os
import shutil
import hashlib
import random
from pathlib import Path

def get_file_md5(file_path):
    """Compute MD5 checksum for a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def prepare_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42
):
    """
    Deduplicates raw images to prevent data leakage and splits dataset into 
    train (70%), validation (10%), and test (20%) sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Classes expected
    classes = ["Normal", "Pothole"]
    
    print("=" * 60)
    print("STARTING DATASET PREPARATION & DEDUPLICATION")
    print("=" * 60)
    print(f"Source Directory: {source_path}")
    print(f"Output Directory: {output_path}")
    print(f"Split Ratio - Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    print(f"Random Seed: {seed}")
    
    # Create target directories
    for split in ["train", "validation", "test"]:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
            
    summary = {}
    
    for cls in classes:
        cls_dir = source_path / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Source folder not found: {cls_dir}")
            
        all_files = [f for f in cls_dir.glob("*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"\nProcessing class '{cls}': Found {len(all_files)} raw files.")
        
        # Deduplication by MD5
        seen_hashes = set()
        unique_files = []
        duplicates = 0
        
        for f in all_files:
            file_hash = get_file_md5(f)
            if file_hash in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(file_hash)
                unique_files.append(f)
                
        print(f"  Unique images: {len(unique_files)} (Removed {duplicates} duplicate images to prevent data leakage)")
        
        # Shuffle deterministically
        random.shuffle(unique_files)
        
        total_unique = len(unique_files)
        train_count = int(total_unique * train_ratio)
        val_count = int(total_unique * val_ratio)
        # Remaining goes to test to avoid rounding off discrepancies
        test_count = total_unique - train_count - val_count
        
        train_files = unique_files[:train_count]
        val_files = unique_files[train_count:train_count + val_count]
        test_files = unique_files[train_count + val_count:]
        
        summary[cls] = {
            "total_raw": len(all_files),
            "duplicates_removed": duplicates,
            "total_unique": total_unique,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        }
        
        # Copy files to destination
        for split_name, files_list in [("train", train_files), ("validation", val_files), ("test", test_files)]:
            dest_dir = output_path / split_name / cls
            # Clean destination directory if already has files
            for existing in dest_dir.glob("*"):
                existing.unlink()
                
            for idx, f in enumerate(files_list):
                dest_file = dest_dir / f"{cls}_{idx:04d}{f.suffix.lower()}"
                shutil.copy2(f, dest_file)
                
    # Also save sample images for Streamlit testing
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    for cls in classes:
        test_cls_dir = output_path / "test" / cls
        first_test_img = next(test_cls_dir.glob("*"), None)
        if first_test_img:
            shutil.copy2(first_test_img, samples_dir / f"sample_{cls.lower()}.jpg")
            
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    for cls, stats in summary.items():
        print(f"Class: {cls:10s} | Total Raw: {stats['total_raw']:4d} | Duplicates: {stats['duplicates_removed']:3d} | Train: {stats['train']:4d} | Val: {stats['val']:4d} | Test: {stats['test']:4d}")
        
    return summary

if __name__ == "__main__":
    source_dataset = r"d:\potholedetection\Normal pothole dataset\Neha Pothole and normal dataset"
    target_dataset = r"d:\potholedetection\dataset"
    prepare_dataset(source_dataset, target_dataset)

