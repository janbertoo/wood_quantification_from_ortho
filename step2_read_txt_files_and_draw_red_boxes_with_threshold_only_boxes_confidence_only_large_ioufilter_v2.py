import rasterio
import numpy as np
import cv2
from osgeo import gdal
from multiprocessing import Pool
import os

output_path = 'detections'

orthos = [
    #'ortho1.tif',
    #'ortho2.tif',
    'VDN2024_clipped_3.tif',
]

def calculate_iou(box1, box2):
    """Calculate Intersection Over Union (IOU) between two bounding boxes."""
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the intersection
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate the areas of both rectangles
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Calculate the union
    union_area = box1_area + box2_area - inter_area

    # Calculate IOU
    if union_area == 0:
        return 0  # Avoid division by zero
    iou = inter_area / union_area
    return iou

def draw_bounding_boxes_on_transparent_background(orthomosaic_path, detections_txt_path, output_bbox_path, filtered_txt_path, conf_threshold):
    # Open the orthomosaic using rasterio to get dimensions
    with rasterio.open(orthomosaic_path) as src:
        width = src.width
        height = src.height
        transform = src.transform

    # Create a transparent background (RGBA)
    bbox_image = np.zeros((height, width, 4), dtype=np.uint8)

    # Read detections from the text file
    with open(detections_txt_path, 'r') as f:
        detections = f.readlines()

    # List to store processed bounding boxes and filtered detections
    boxes_conf = []  # Store tuples of (box, confidence, detection)
    filtered_detections = []

    # Process bounding boxes and filter based on IOU
    for detection in detections:
        parts = detection.strip().split(',')
        if len(parts) != 9:
            continue
        try:
            detection_id, conf, x_center, y_center, x_min_real, y_min_real, x_max_real, y_max_real, length = map(float, parts)

            # Only process detections above the confidence threshold
            if conf < conf_threshold or length < 1:
                continue

            # Convert real-world coordinates to pixel coordinates
            x_min, y_min = ~transform * (x_min_real, y_min_real)
            x_max, y_max = ~transform * (x_max_real, y_max_real)

            # Ensure coordinates are within the image bounds
            x_min = np.clip(int(x_min), 0, width - 1)
            y_min = np.clip(int(y_min), 0, height - 1)
            x_max = np.clip(int(x_max), 0, width - 1)
            y_max = np.clip(int(y_max), 0, height - 1)

            # New box in pixel coordinates
            new_box = (x_min, y_min, x_max, y_max)

            # Check IOU with existing boxes and keep the one with the highest confidence
            keep = True
            for i, (existing_box, existing_conf, existing_detection) in enumerate(boxes_conf):
                iou = calculate_iou(new_box, existing_box)
                if iou > 0.6:
                    if conf > existing_conf:
                        # Replace existing box with the new one as it has higher confidence
                        boxes_conf[i] = (new_box, conf, detection)
                    keep = False
                    break

            if keep:
                # Append the new box, confidence, and detection to the list
                boxes_conf.append((new_box, conf, detection))

        except ValueError as ve:
            print(f"Error parsing line: {detection}. Error: {ve}")
            continue

    # Now that we have filtered the boxes, draw the remaining ones on the transparent background
    for box, conf, detection in boxes_conf:
        x_min, y_min, x_max, y_max = box

        # Draw rectangle on the transparent background with red color (RGBA format)
        cv2.rectangle(bbox_image, (x_min, y_min), (x_max, y_max), (255, 0, 0, 255), 2)  # Red color with thickness 2

        # Extract detection ID from the detection string
        detection_id = int(detection.split(',')[0])

        # Draw ID number with confidence score above the bounding box
        text = f"{detection_id}: {conf:.3f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = int(x_min + (x_max - x_min) / 2 - text_size[0] / 2)
        text_y = int(y_min - 5)
        cv2.putText(bbox_image, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0, 255), 2)

        # Append to filtered detections
        filtered_detections.append(detection)

    # Convert bbox_image to CHW layout for saving
    bbox_image = np.moveaxis(bbox_image, -1, 0)  # Convert from HWC to CHW layout

    # Create metadata for the new image
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 4,
        'dtype': 'uint8',
        'crs': src.crs,
        'transform': transform,
        'compress': 'lzw',
        'tiled': 'yes',
        'blockxsize': 256,
        'blockysize': 256
    }

    # Save the transparent background with red bounding boxes using GDAL
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_bbox_path, width, height, 4, gdal.GDT_Byte, ['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    dst_ds.SetGeoTransform(transform.to_gdal())
    dst_ds.SetProjection(src.crs.to_wkt())

    for i in range(4):
        dst_ds.GetRasterBand(i+1).WriteArray(bbox_image[i])

    dst_ds.FlushCache()
    dst_ds = None

    print(f"Saved output bounding boxes with transparency to {output_bbox_path}")

    # Save filtered detections to a new text file
    with open(filtered_txt_path, 'w') as f:
        for detection in filtered_detections:
            f.write(detection)

    print(f"Saved filtered detections to {filtered_txt_path}")

# Example usage:
def process(ortho):
    for percentage in [10,20,30,40,50,60,70,80,90]:
        draw_bounding_boxes_on_transparent_background(ortho, os.path.join(output_path,ortho.split('.tif')[0]+'_detections_conf_10perc_or.txt'), os.path.join(output_path,ortho.split('.tif')[0]+'_with_boxes_'+str(percentage)+'perc.tif'), os.path.join(output_path,ortho.split('.tif')[0]+'_detections_conf_'+str(percentage)+'perc.txt'), conf_threshold=percentage/100)

if __name__ == '__main__':
    with Pool(6) as p:
        print(p.map(process, orthos))
