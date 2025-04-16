import easyocr
import cv2
import matplotlib.pyplot as plt

def process_invoice(image_path):
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
    results = reader.readtext(image_path)  # Read text from the image

    extracted_text = []
    image = cv2.imread(image_path)

    for (bbox, text, prob) in results:
        extracted_text.append(text)
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label text

    extracted_text_str = "\n".join(extracted_text)

    print("Extracted Text:")
    print(extracted_text_str)  # Print extracted text in terminal
    print(f"OCR Processing Done for: {image_path}")  # Confirm processing

    # Show image with bounding boxes
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return extracted_text_str

# Run the script with your image path
image_path = "/Users/atharvakhot/Downloads/X/uploads/sample_invoice.jpg"
process_invoice(image_path)