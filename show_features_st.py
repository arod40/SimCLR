import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
from data_aug.square_pad import SquarePad


def load_encoder(path: str, device: str = "cuda"):
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    model = torchvision.models.resnet18(pretrained=False, num_classes=4).to(device)
    checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        if k.startswith("backbone."):
            if k.startswith("backbone") and not k.startswith("backbone.fc"):
                # remove prefix
                state_dict[k[len("backbone.") :]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ["fc.weight", "fc.bias"]

    model.fc = Identity()
    model.eval()

    return model


def load_decoder(path: str, device="cuda"):
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 96 * 96),
        nn.Sigmoid(),
    ).to(device)

    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def load_image():
    pil_image = Image.open(image)
    transform = transforms.Compose(
        [
            SquarePad(padding_mode="edge"),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(pil_image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        encoded = state.encoder(image_tensor)
        st.text(encoded.shape)


state = st.session_state

with st.sidebar:
    feature = st.selectbox("Feature", range(512))
    value = st.slider("Value", 0.0, 20.0, 1.0, 0.00001)

if "encoder" not in state:
    state.encoder = load_encoder(
        "runs/May11_17-17-03_LAPTOP-HL6IE8PG/checkpoint_0001.pth.tar", device="cuda"
    )

if "decoder" not in state:
    state.decoder = load_decoder("decoder.pth.tar", device="cuda")

image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if image is not None:
    pil_image = Image.open(image)
    transform = transforms.Compose(
        [
            SquarePad(padding_mode="edge"),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(pil_image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        encoded = state.encoder(image_tensor)
        encoded[0, feature] = value
        decoded = state.decoder(encoded)

        st.text(encoded.shape)

    if image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.text("Original")
            st.image(image, width=256)

        with col2:
            st.text("Reconstructed")
            st.image(
                Image.fromarray(
                    (decoded.view(96, 96).cpu().numpy() * 256).astype(int)
                ).convert("RGB"),
                width=256,
            )
