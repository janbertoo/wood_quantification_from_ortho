import rasterio
import numpy as np
from ultralytics import YOLO
import cv2
import math
import os

output_path = 'detections'

orthos = [
    #'ortho1.tif',
    #'ortho2.tif',
    'VDN2024_clipped_3.tif',
]

if not os.path.exists(output_path):
    os.makedirs(output_path) 

def detect_wood_in_orthomosaic(model_path, orthomosaic_path, output_txt_path, conf_threshold):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    all_data = []

    # Open the orthomosaic using rasterio
    with rasterio.open(orthomosaic_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        
        # Function to get coordinates from pixel indices
        def pixel_to_coords(x, y):
            return transform * (x, y)
        
        # Initialize lists to store final detections
        all_detections = []
        detection_id = 1

        def process_tile(tile, x_offset, y_offset):
            nonlocal detection_id
            # Ensure the tile is 960x960 by padding if necessary
            if tile.shape[1] < 960 or tile.shape[2] < 960:
                tile = np.pad(tile, 
                              ((0, 0), 
                               (0, max(0, 960 - tile.shape[1])), 
                               (0, max(0, 960 - tile.shape[2]))), 
                              mode='constant', 
                              constant_values=0)
            
            # Check the number of channels and convert to RGB if necessary
            if tile.shape[0] == 4:
                tile = tile[:3, :, :]  # Take only the first 3 channels
            
            # Convert to the required format for YOLO
            tile = np.moveaxis(tile, 0, -1)
            
            # Run the model with the confidence threshold
            results = model(tile, conf=conf_threshold)
            for result in results:
                for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                    x_min, y_min, x_max, y_max = box.tolist()
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    
                    # Check if the center is within the 480x480 central region of the 960x960 tile
                    if 240 <= x_center <= 720 and 240 <= y_center <= 720:
                        # Adjust coordinates relative to the orthomosaic
                        x_min += x_offset
                        y_min += y_offset
                        x_max += x_offset
                        y_max += y_offset
                        x_center += x_offset
                        y_center += y_offset
                        
                        # Convert pixel coordinates to map coordinates
                        x_center_coord, y_center_coord = pixel_to_coords(x_center, y_center)
                        x_min_coord, y_min_coord = pixel_to_coords(x_min, y_min)
                        x_max_coord, y_max_coord = pixel_to_coords(x_max, y_max)
                        
                        # Calculate diagonal length in meters
                        diagonal_length = math.sqrt((x_max_coord - x_min_coord)**2 + (y_max_coord - y_min_coord)**2)
                        
                        # Store the detection with ID and confidence score
                        all_detections.append(
                            f"{int(detection_id)},{conf},{x_center_coord},{y_center_coord},{x_min_coord},{y_min_coord},{x_max_coord},{y_max_coord},{diagonal_length}"
                        )
                        detection_id += 1

        # Generate tiles and process them
        #for y in range(0, height, 960):
            #for x in range(0, width, 960):
                #window = rasterio.windows.Window(x, y, min(960, width - x), min(960, height - y))
                #tile = src.read(window=window)
                #process_tile(tile, x, y)

                # Generate tiles and process them
        for y in range(0, height, 960):
            for x in range(0, width, 960):
                window = rasterio.windows.Window(x, y, min(960, width - x), min(960, height - y))
                tile = src.read(window=window)
                process_tile(tile, x, y)

        for y in range(0, height, 960):
            for x in range(480, width, 960):
                window = rasterio.windows.Window(x, y, min(960, width - x), min(960, height - y))
                tile = src.read(window=window)
                process_tile(tile, x, y)

        for y in range(480, height, 960):
            for x in range(0, width, 960):
                window = rasterio.windows.Window(x, y, min(960, width - x), min(960, height - y))
                tile = src.read(window=window)
                process_tile(tile, x, y)
        
        for y in range(480, height, 960):
            for x in range(480, width, 960):
                window = rasterio.windows.Window(x, y, min(960, width - x), min(960, height - y))
                tile = src.read(window=window)
                process_tile(tile, x, y)

        # Save all detections to a text file
        with open(output_txt_path, 'w') as f:
            for detection in all_detections:
                f.write(detection + '\n')



conf_threshold=0.1




def draw_bounding_boxes_on_orthomosaic(orthomosaic_path, detections_txt_path, output_orthomosaic_path):
    # Open the orthomosaic using rasterio
    with rasterio.open(orthomosaic_path) as src:
        orthomosaic = src.read()
        transform = src.transform

        # Convert the orthomosaic to a format suitable for OpenCV (HWC layout)
        if orthomosaic.shape[0] == 4:
            orthomosaic = orthomosaic[:3, :, :]  # Discard alpha channel if present
        orthomosaic = np.moveaxis(orthomosaic, 0, -1)  # Convert from CHW to HWC layout
        
        # Ensure the image is contiguous and in the right type for OpenCV
        orthomosaic = np.ascontiguousarray(orthomosaic, dtype=np.uint8)

        print("Orthomosaic shape (HWC):", orthomosaic.shape)  # Debug statement

        # Read detections from the text file
        with open(detections_txt_path, 'r') as f:
            detections = f.readlines()

        # Draw bounding boxes and ID numbers on the orthomosaic
        for detection in detections:
            parts = detection.strip().split(',')
            if len(parts) != 9:
                print(f"Skipping malformed line: {detection}")
                continue
            try:
                detection_id, _, x_center, y_center, x_min_real, y_min_real, x_max_real, y_max_real, _ = map(float, parts)

                # Convert real-world coordinates to pixel coordinates
                x_min, y_min = ~transform * (x_min_real, y_min_real)
                x_max, y_max = ~transform * (x_max_real, y_max_real)

                # Ensure coordinates are within the image bounds
                x_min = np.clip(int(x_min), 0, orthomosaic.shape[1] - 1)
                y_min = np.clip(int(y_min), 0, orthomosaic.shape[0] - 1)
                x_max = np.clip(int(x_max), 0, orthomosaic.shape[1] - 1)
                y_max = np.clip(int(y_max), 0, orthomosaic.shape[0] - 1)

                # Draw rectangle on the orthomosaic with red color (BGR format)
                cv2.rectangle(orthomosaic, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Red color with thickness 2

                # Draw ID number next to the bounding box
                cv2.putText(orthomosaic, str(int(detection_id)), (int(x_min), int(y_min) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except ValueError as ve:
                print(f"Error parsing line: {detection}. Error: {ve}")
                continue

        # Convert orthomosaic back to the original format (CHW layout) for saving
        orthomosaic = np.moveaxis(orthomosaic, -1, 0)  # Convert from HWC to CHW layout
        meta = src.meta
        meta.update({"count": orthomosaic.shape[0], "dtype": "uint8"})

        with rasterio.open(output_orthomosaic_path, 'w', **meta) as dst:
            dst.write(orthomosaic)

        print(f"Saved output orthomosaic with red bounding boxes and IDs to {output_orthomosaic_path}")




def apply_nms_and_save(detections, output_txt_path, iou_threshold):
    # Function to compute IoU
    def compute_iou(box1, box2):
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        xi1 = max(x_min1, x_min2)
        yi1 = max(y_min1, y_min2)
        xi2 = min(x_max1, x_max2)
        yi2 = min(y_max1, y_max2)
        inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
        
        box1_area = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
        box2_area = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area

    # Sort detections by confidence score in descending order
    detections = sorted(detections, key=lambda x: x[-1], reverse=True)

    # Apply Non-Maximum Suppression
    final_detections = []
    while detections:
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        detections = [
            det for det in detections
            if compute_iou(best_detection[2:6], det[2:6]) < iou_threshold
        ]

    # Save filtered detections to a text file
    with open(os.path.join(output_path,output_txt_path, 'w')) as f:
        for detection in final_detections:
            f.write(','.join(map(str, detection)) + '\n')

for ortho in orthos:
	txtfile = os.path.join(output_path,ortho.replace('.tif', '_detections_conf_'+str(int(conf_threshold*100))+'perc_or.txt'))
	ortho_drawn = ortho.replace('.tif','_with_red_boxes.tif')
	detect_wood_in_orthomosaic('v10_train_wood_from_ortho_27aug_small/weights/best.pt', ortho, txtfile, conf_threshold=conf_threshold)
	#draw_bounding_boxes_on_orthomosaic(ortho, txtfile, ortho_drawn)
