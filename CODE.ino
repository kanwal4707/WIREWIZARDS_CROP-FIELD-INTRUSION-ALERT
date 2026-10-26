from ultralytics import YOLO

# Load the YOLO AI model
model = YOLO("yolo26n.pt")

# Animal classes supported by this model
animals = [
    "dog", "cat", "cow", "sheep",
    "horse", "bird", "bear", "elephant",
    "zebra", "giraffe"
]

print("Crop-Field Intrusion Detection Started!")
print("Press Q to stop.\n")

# Start laptop webcam
for result in model.predict(
    source=0,
    show=True,
    stream=True,
    conf=0.4
):
    detected = set()

    for box in result.boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]

        if name == "person":
            detected.add("HUMAN DETECTED")

        elif name in animals:
            detected.add(f"ANIMAL DETECTED: {name.upper()}")

    for message in sorted(detected):
        print(message)
