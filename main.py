from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

classes = ['paper', 'rock', 'scissors']

check_hand = FastAPI()

class CheckHand(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    )

  def forward(self, x):
        return self.second(self.net(x))

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheckHand()
model.load_state_dict(torch.load('hands.pth', map_location=device))
model.to(device)
model.eval()


@check_hand.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        if not image_data:
            raise HTTPException(status_code=404, detail="No image")

        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {"Answer": classes[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(check_hand, host="127.0.0.1", port=8000)