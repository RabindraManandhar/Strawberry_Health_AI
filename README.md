# Strawberry_Health_AI

### 1. Model-1 Default training for 5 epochs => train


### 2. Model-2 Fine-tuned for 20 epochs in original dataset => runs/detect/train2
20 epochs completed in 2.465 hours.

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

Model summary (fused): 186 layers, 2,685,733 parameters, 0 gradients, 6.8 GFLOPs
                
Speed: 0.4ms preprocess, 27.1ms inference, 0.0ms loss, 0.6ms postprocess per image

### 3. Model-3 Fine-tuned for 20 epochs in dataset with grayscale images => runs/detect/train3

20 epochs completed in 0.147 hours.

| Class                  | Images | Instances | P     | R     | mAP50 | mAP50-95 |
|------------------------|--------|-----------|-------|-------|-------|----------|
| all                    | 308    | 754       | 0.575 | 0.636 | 0.632 | 0.448    |
| Angular Leafspot        | 43     | 52        | 0.84 | 0.731 | 0.782 | 0.529    |
| Anthracnose Fruit Rot   | 13     | 20        | 0.455 | 0.15 | 0.215 | 0.0772    |
| Blossom Blight          | 29     | 44        | 0.73 | 1 | 0.956 | 0.678    |
| Gray Mold              | 77     | 108       | 0.63 | 0.583 | 0.596 | 0.353    |
| Leaf Spot              | 71     | 257       | 0.65 | 0.798 | 0.798 | 0.664    |
| Powdery Mildew Fruit    | 12     | 18        | 0.227 | 0.333 | 0.295 | 0.18    |
| Powdery Mildew Leaf     | 63     | 255       | 0.492 | 0.859 | 0.783 | 0.652    |

Model summary (fused): 186 layers, 2,685,733 parameters, 0 gradients, 6.8 GFLOPs

Speed: 0.2ms preprocess, 1.2ms inference, 0.0ms loss, 0.9ms postprocess per image

### 4. Model-4 Fine-tuned for 50 epochs in original dataset => runs/detect/train4

50 epochs completed in 0.365 hours.

| Class                  | Images | Instances | P     | R     | mAP50 | mAP50-95 |
|------------------------|--------|-----------|-------|-------|-------|----------|
| all                    | 308    | 754       | 0.764 |     0.836  |    0.847  |    0.618    |
| Angular Leafspot        | 43     | 52        | 0.978 |      0.84  |    0.889  |    0.619    |
| Anthracnose Fruit Rot   | 13     | 20        | 0.689 |       0.7  |   0.725   |  0.409    |
| Blossom Blight          | 29     | 44        | 0.839 |         1  |  0.989    |   0.824    |
| Gray Mold              | 77     | 108       | 0.794 |     0.704  |    0.797  |    0.491    |
| Leaf Spot              | 71     | 257       | 0.732 |     0.868  |    0.896  |   0.772    |
| Powdery Mildew Fruit    | 12     | 18        | 0.716 |      0.84  |     0.783 |     0.472    |
| Powdery Mildew Leaf     | 63     | 255       | 0.598 |     0.898  |     0.852 |     0.737    |

Model summary (fused): 186 layers, 2,685,733 parameters, 0 gradients, 6.8 GFLOPs

Speed: 0.2ms preprocess, 1.2ms inference, 0.0ms loss, 0.9ms postprocess per image