import json
import os

mot20_dir = '/mot20_dir'
output_json_path = '/output.json'

category_names = ['ceratium furca', 'ceratium fucus', 'ceratium trichoceros', 'chaetoceros curvisetus', 'cladocera',
              'copepoda1', 'copepoda2', 'coscinodiscus', 'curve thalassiosira', 'guinardia delicatulad', 'helicotheca',
              'lauderia cleve', 'skeletonema', 'thalassionema nitzschioides', 'thalassiosira nordenskioldi', 'tintinnid',
              'sanguinea', 'Thalassiosira rotula', 'Protoperidinium', 'Eucampia zoodiacus', 'Guinardia striata']

coco_format = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i+1, "name": name} for i, name in enumerate(category_names)]
}

img_dir = os.path.join(mot20_dir, "val")
ann_path = os.path.join(mot20_dir, "val.txt")

image_id = 0
for img_name in sorted(os.listdir(img_dir)):
    if img_name.endswith('.jpg'):
        image_id += 1
        image_path = os.path.join(img_dir, img_name)
        coco_format["images"].append({
            "file_name": img_name,
            "id": image_id,
            "width": 640,
            "height": 640
        })

annotation_id = 0
with open(ann_path, 'r') as file:
    for line in file:
        frame_id, track_id, x, y, w, h, conf, category_id, _ = map(float, line.strip().split(','))
        if conf == 0:
            continue
        annotation_id += 1
        image_id = int(frame_id)
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "bbox": [x, y, w, h],
            "category_id": int(category_id),
            "iscrowd": 0,
            "area": w * h,
        })

with open(output_json_path, 'w') as f:
    json.dump(coco_format, f)

