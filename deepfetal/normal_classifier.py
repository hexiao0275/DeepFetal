import os
import torch
from PIL import Image
from torchvision import transforms
import json
from .model import convnext_tiny
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Custom dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, image_path

def get_all_image_paths(folder):
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def classify_images_dataloader(
    dataloader,
    model,
    device,
    class_labels
):
    classification_results = {}

    for batch in tqdm(dataloader, desc="Batch Processing", ncols=100):
        images, image_paths = batch
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            top_probs, top_indices = torch.max(probs, dim=1)

            for i in range(len(image_paths)):
                predicted_index = int(top_indices[i].cpu().numpy())
                predicted_label = class_labels.get(str(predicted_index), "Unknown")

                # Get per-class probabilities
                probabilities = {class_labels[str(idx)]: float(prob) for idx, prob in enumerate(probs[i].cpu())}
                probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

                classification_results[image_paths[i]] = {
                    "probabilities": probabilities,
                    "predicted_label": predicted_label
                }

    return classification_results

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # num_classes = 41
    num_classes = args.normal_cls["model"]["num_classes"]
    # img_size = 224
    img_size = args.normal_cls["dataloader"]["img_size"]
    # batch_size = 512
    batch_size = args.normal_cls["dataloader"]["batch_size"]  # Tune this based on available GPU memory
    num_workers = args.normal_cls["dataloader"]["num_workers"]
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = convnext_tiny(num_classes=num_classes).to(device)
    # weights_path = "./weights/2_3_cls_model.pth"
    weights_path = args.normal_cls["model"]["weights_path"]
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # class_labels_path = "./class_indices_41.json"
    class_labels_path = args.normal_cls["model"]["class_indices_path"]
    with open(class_labels_path, 'r') as f:
        class_labels = json.load(f)

    # base_image_folder = "./copied_reports_only_eval_224"
    base_image_folder = args.image_root
    image_paths = get_all_image_paths(base_image_folder)

    dataset = ImageDataset(image_paths, data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    results = classify_images_dataloader(
        dataloader,
        model,
        device,
        class_labels
    )

    results = {
        k.replace(args.root_mapping[1], args.root_mapping[0], 1) : v
        for k, v in results.items()
    }

    # with open("2_1_classification_results_with_probabilities_224_eval.json", 'w') as f:
    with open(args.normal_cls["out_json"], 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Classification complete. Results saved to {args.normal_cls['out_json']}.")

if __name__ == '__main__':
    main()
