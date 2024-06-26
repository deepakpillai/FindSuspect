import cv2
from facenet_pytorch import InceptionResnetV1,MTCNN, extract_face
from PIL import Image
import torchvision.transforms as transforms
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained Inception ResNet model
#vggface2 is better then casia-webface in-terms of accuracy
resnet_model = InceptionResnetV1(pretrained='vggface2').eval()
# Initialize MTCNN for face detection
mtcnn = MTCNN()
# image to a Torch tensor
transform = transforms.Compose([
    transforms.ToTensor() 
])

findx = Image.open("findx.png").convert("RGB") 
# Detect face
boxes, _ = mtcnn.detect(findx)
# Ensure at least one face is detected
if boxes is not None:
    # Extract and save the face
    extract_face(findx, box=boxes[0], image_size=255, save_path="x.png")
else:
    print("No faces detected.")

cap = cv2.VideoCapture('input/videoplayback (4).mp4')
count = 0
suspect = Image.open("x.png")
suspect_tensor = transform(suspect).unsqueeze(0)  # Add batch dimension
suspect_embedding = resnet_model(suspect_tensor).detach()

while cap.isOpened():
    ret,frame = cap.read()

    boxes, confidance = mtcnn.detect(frame)
    if boxes is not None:
        i = 0
        for box in boxes:
            start_point = (int(box[0]), int(box[1])) 
            end_point = (int(box[2]), int(box[3])) 
            # Blue color in BGR 
            color = (255, 0, 0) 
            # Line thickness of 2 px 
            thickness = 2
            try:
                # Crop face using the bounding box
                face = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
                bigger_image = cv2.resize(face, (255, 255))
                img = Image.fromarray(bigger_image)
                img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
                img_embedding = resnet_model(img_tensor).detach()
                similarity = cosine_similarity(suspect_embedding, img_embedding)
                if similarity[0][0] > 0.75:

                    color = (0, 0, 255) 
                    thickness = 2
                    # font 
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    # fontScale 
                    fontScale = 1
                    start_point_for_text = (int(box[0]) - 10, int(box[1]) - 10) 
                    cv2.putText(frame, 'Mr.X', start_point_for_text, font, fontScale, color, thickness, cv2.LINE_AA) 
                cv2.rectangle(frame, start_point, end_point, color, thickness) 
                
            except Exception as e:
                print(e)

    cv2.imshow('Video feed', frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # destroy all opened windows