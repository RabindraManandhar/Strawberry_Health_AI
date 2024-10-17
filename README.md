# Strawberry_Health_AI

### 1. Model-1 Default training with 5 epochs => train


### 2. Model-2 Fine-tune with 20 epochs => train2
20 epochs completed in 2.465 hours.

Model summary (fused): 186 layers, 2,685,733 parameters, 0 gradients, 6.8 GFLOPs

| Class                  | Images | Instances | P     | R     | mAP50 | mAP50-95 |
|------------------------|--------|-----------|-------|-------|-------|----------|
| all                    | 308    | 754       | 0.709 | 0.722 | 0.781 | 0.565    |
| Angular Leafspot        | 43     | 52        | 0.885 | 0.731 | 0.848 | 0.539    |
| Anthracnose Fruit Rot   | 13     | 20        | 0.789 | 0.300 | 0.640 | 0.353    |
| Blossom Blight          | 29     | 44        | 0.751 | 1.000 | 0.969 | 0.774    |
| Gray Mold              | 77     | 108       | 0.774 | 0.667 | 0.752 | 0.462    |
| Leaf Spot              | 71     | 257       | 0.737 | 0.844 | 0.862 | 0.736    |
| Powdery Mildew Fruit    | 12     | 18        | 0.513 | 0.611 | 0.549 | 0.370    |
| Powdery Mildew Leaf     | 63     | 255       | 0.511 | 0.902 | 0.847 | 0.722    |
                
Speed: 0.4ms preprocess, 27.1ms inference, 0.0ms loss, 0.6ms postprocess per image

Results saved to /home/aiot-garage/Strawberry_Health_AI/runs/detect/train2

Model fine-tuned for 20 epochs.