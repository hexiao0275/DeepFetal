import os
import cv2
import time
import json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm

def batch_classify_images(input_folder, model_path,out_json, conf_threshold=0.5, class_conf_threshold=0.85):
    """
    Recursively classify images in subfolders with a YOLO model and save JSON only.

    Args:
        input_folder: input image folder path, including subfolders
        model_path: YOLO model path
        conf_threshold: inference confidence threshold, default 0.5
        class_conf_threshold: classification confidence threshold, default 0.85
    """
    
    # Create output folder
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    
    # Recursively collect image files
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No image files were found in {input_folder} or its subfolders.")
        return
    
    print(f"Found {len(image_paths)} images, starting classification...")
    
    # Initialize result stats
    classification_results = {
        "summary": defaultdict(int),
        "details": [],
        "uncertain_images": [],
        "class_names": model.names
    }
    
    # Track processing time
    start_time = time.time()
    
    # Progress bar
    progress_bar = tqdm(image_paths, desc="Classification progress", unit="image")
    
    # Batch process images
    for img_path in progress_bar:
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Unable to read image: {img_path}")
                classification_results["details"].append({
                    "original_path": os.path.relpath(img_path, input_folder),
                    "error": "Unable to read image"
                })
                continue
            
            # Record source path information
            relative_path = os.path.relpath(img_path, input_folder)
            filename = os.path.basename(img_path)
            
            # Run model inference
            results = model(img, conf=conf_threshold)
            
            # Read classification results
            for result in results:
                if hasattr(result, 'probs'):  # classification task
                    # Collect confidence scores for all classes
                    all_probs = result.probs.data.cpu().numpy()
                    top5_indices = np.argsort(all_probs)[-5:][::-1]
                    
                    # Build full confidence dict
                    confidences = {
                        model.names[i]: float(all_probs[i]) 
                        for i in range(len(model.names))
                    }
                    
                    # Pick top-1 class
                    top1_index = result.probs.top1
                    top1_conf = result.probs.top1conf.item()
                    class_name = model.names[top1_index]

                    # Check threshold
                    if top1_conf >= class_conf_threshold:
                        status = class_name
                        classification_results["summary"][class_name] += 1
                    else:
                        status = "uncertain"
                        classification_results["uncertain_images"].append(relative_path)

                    # Save classification result with full confidence scores
                    classification_results["details"].append({
                        "original_path": relative_path,
                        "classified_as": status,
                        "top1_confidence": float(top1_conf),
                        "all_confidences": confidences,
                        "top5_classes": [
                            {"class": model.names[idx], "confidence": float(all_probs[idx])} 
                            for idx in top5_indices
                        ]
                    })

                    # Update progress bar
                    progress_bar.set_postfix({
                        "class": status,
                        "conf": f"{top1_conf:.2f}"
                    })
                
        except Exception as e:
            print(f"Error while processing image {img_path}: {str(e)}")
            classification_results["details"].append({
                "original_path": os.path.relpath(img_path, input_folder),
                "error": str(e)
            })
    
    # Close progress bar
    progress_bar.close()
    
    # Compute total runtime
    total_time = time.time() - start_time
    avg_time = total_time / len(image_paths) if len(image_paths) > 0 else 0
    
    # Add summary stats
    classification_results["statistics"] = {
        "total_images": len(image_paths),
        "processed_images": len([x for x in classification_results["details"] if "error" not in x]),
        "failed_images": len([x for x in classification_results["details"] if "error" in x]),
        "total_time_seconds": total_time,
        "average_time_per_image": avg_time
    }
    
    # Save JSON output
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(classification_results, f, ensure_ascii=False, indent=4)
    
    print("\nBatch classification finished.")
    print(f"Total images: {len(image_paths)}")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.2f} seconds")
    print(f"Classification output saved to: {out_json}")

def main(args):
    # input_folder = "./copied_reports_only_eval_224"
    input_folder = args.image_root
    # output_folder = "./eval_classification_results"
    # output_folder = args.early_cls["output_folder"]
    # model_path =  "./best.pt"
    model_path =  args.early_cls["model"]["weights_path"]
    conf_threshold = args.early_cls["model"]["conf_threshold"]
    class_conf_threshold = args.early_cls["model"]["class_conf_threshold"]
    out_json = args.early_cls["out_json"]
    batch_classify_images(input_folder, model_path, out_json, conf_threshold, class_conf_threshold)

if __name__ == "__main__":
    raise SystemExit("Run this module through deepfetal.process.")
