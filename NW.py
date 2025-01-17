#pip install ultralytics opencv-python geopandas shapely rasterio

import os
from ultralytics import YOLO
from pathlib import Path
import cv2
import geopandas as gpd
from shapely.geometry import box  # Correctly import box for polygon creation

def extract_bounderies(geotiff_path):
    """Extract the lowest x and y (real-world coordinates) from a GeoTIFF file."""
    import rasterio
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        return bounds.left, (bounds.top), bounds.right, (bounds.bottom)   # Real-world origin (lowest_x, lowest_y)

def predict_and_label(model_path, input_folder="Imageries", output_folder="Output"):
    """
    Predict objects in images, save images, text files, and shapefiles (one per class).
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the YOLO model with the trained weights
    model = YOLO(model_path)

    # Define color mapping for classes
    class_colors = {
        0: (0, 0, 255),  # Class 0 in red (BGR format)
        1: (0, 255, 0),  # Class 1 in green (BGR format)
        # Add more classes if needed
    }

    # Initialize a dictionary to hold bounding boxes for each class
    shapefile_data = {cls: [] for cls in class_colors.keys()}

    # Iterate through all images in the input folder
    for image_path in Path(input_folder).glob("*.*"):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
            print(f"Processing: {image_path}")

            lowest_x, lowest_y, highest_x, highest_y = extract_bounderies(image_path)
            print(f"X, Y coordinate bounderies (GeoTIFF): {lowest_x,highest_x} to {-lowest_y,-highest_y}")

            # Predict objects in the image
            results = model.predict(
                source=str(image_path),
                conf=0.0005,  # Confidence threshold
                save=False,  # Disable automatic save to avoid subfolders
                save_txt=False  # Disable automatic save for text
            )

            # Define class labels
            class_labels = {
                0: "Boat",
                1: "Lobster Buoy",
                # Add more classes if needed
            }

            # Load the image
            img = cv2.imread(str(image_path))

            # Get dimensions of the annotated image
            image_height, image_width = img.shape[:2]

            lowest_x_Ann, lowest_y_Ann, highest_x_Ann, highest_y_Ann = 0, 0, image_width-1, image_height-1
            
            print(f"X, Y coordinate bounderies (Annotated): {lowest_x_Ann,highest_x_Ann} to {lowest_y_Ann,highest_y_Ann}")

            # Adjust bounding boxes to real-world coordinates
            for box_object in results[0].boxes:
                cls = int(box_object.cls.item())  # Class index
                x1, y1, x2, y2 = map(int, box_object.xyxy[0].tolist())  # Bounding box coordinates (pixel)
                
                # Scaling factor
                sx = (highest_x-lowest_x)/(highest_x_Ann-lowest_x_Ann)
                sy = (highest_y-lowest_y)/(highest_y_Ann-lowest_y_Ann)
                
                # Convert pixel coordinates to real-world coordinates
                real_x1, real_y1 = lowest_x + sx * (x1 - lowest_x_Ann), lowest_y + sy * (y1 - lowest_y_Ann)
                real_x2, real_y2 = lowest_x + sx * (x2 - lowest_x_Ann), lowest_y + sy * (y2 - lowest_y_Ann)

                # Append the bounding box and confidence interval as a polygon to the appropriate class
                conf = float(box_object.conf.item())  # Confidence score
                shapefile_data[cls].append((box(real_x1, real_y1, real_x2, real_y2), conf))
                
                # Get the color for the current class (default to white if not defined)
                color = class_colors.get(cls, (255, 255, 255))

                # Draw the bounding box in the original pixel space for visualization
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add the label text with confidence (exclude if you only need the boxes)
                label = class_labels.get(cls, "Unknown")  # Get the label
                text = f"{label}: {conf:.2f}"  # Combine label with confidence
                text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)  # Adjust position
                cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            #print(shapefile_data)
            # Save the manually rendered image
            annotated_image_path = Path(output_folder) / f"{image_path.stem}_predicted{image_path.suffix}"
            cv2.imwrite(str(annotated_image_path), img)

            # Save annotation text file manually
            annotation_txt_path = Path(output_folder) / f"{image_path.stem}.txt"

            with open(annotation_txt_path, "w") as f:
                for box_object in results[0].boxes:
                    cls = int(box_object.cls.item())  # Class index
                    conf = float(box_object.conf.item())  # Confidence score
                    x1, y1, x2, y2 = box_object.xyxy[0].tolist()  # Bounding box coordinates
                    real_x1, real_y1 = lowest_x + sx * (x1 - lowest_x_Ann), lowest_y + sy * (y1 - lowest_y_Ann)
                    real_x2, real_y2 = lowest_x + sx * (x2 - lowest_x_Ann), lowest_y + sy * (y2 - lowest_y_Ann)
                    f.write(f"{cls} {conf:.4f} {real_x1:.4f} {real_y1:.4f} {real_x2:.4f} {real_y2:.4f}\n")

    # Save shapefiles for each class
    for cls, polygons in shapefile_data.items():
        if polygons:
            # Separate the polygons and confidence scores
            polygons, confidences = zip(*polygons)

            # Create a GeoDataFrame including the confidence scores
            gdf = gpd.GeoDataFrame(
                {"geometry": polygons, "class": [cls] * len(polygons), "confidence": confidences},
                crs="EPSG:26919"  # CRS for NAD83 / UTM zone 19N
            )

            # Save the GeoDataFrame as a shapefile
            shapefile_path = Path(output_folder) / f"{image_path.stem}_c_{cls}_bbs.shp"
            gdf.to_file(shapefile_path)

    print(f"Shapefile saved: {output_folder}")

# Predict using model 1
trained_model_file = "yolo11x_model1_173.pt"
predict_and_label(trained_model_file, input_folder="Imageries", output_folder="Output_model1")

# Predict using model 2
trained_model_file = "yolo11x_model2_414.pt" 
predict_and_label(trained_model_file, input_folder="Imageries", output_folder="Output_model2")

