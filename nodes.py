import io
import os
import json
import time
import requests
import numpy as np
import torch
from PIL import Image


FAL_KEY = os.environ.get("FAL_KEY", "")

ASPECT_RATIOS = [
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "21:9",
    "9:21",
    "3:2",
    "2:3",
    "5:4",
]

OUTPUT_FORMATS = ["png", "jpeg", "webp"]

RESOLUTIONS = ["1K", "2K", "4K"]



def _resolve_api_key(fal_api_key=""):
    """Return the API key from the node field, falling back to the env var."""
    key = fal_api_key.strip() if fal_api_key else ""
    if not key:
        key = FAL_KEY
    if not key:
        raise ValueError(
            "No fal.ai API key provided. Either enter it on the node "
            "or set the FAL_KEY environment variable. "
            "Get a key at https://fal.ai/dashboard/keys"
        )
    return key


def _submit_request(endpoint, payload, timeout, api_key):
    """Submit a request to fal.ai and poll until complete."""
    url = f"https://queue.fal.run/{endpoint}"
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if not resp.ok:
        raise RuntimeError(
            f"fal.ai submit failed ({resp.status_code}): {resp.text}"
        )
    queue_data = resp.json()

    request_id = queue_data.get("request_id")
    if not request_id:
        raise RuntimeError(f"No request_id in queue response: {queue_data}")

    # fal.ai app ID is "owner/name" (first two segments); any extra segments
    # (e.g. /edit) are routes within the app and must NOT appear in the
    # status / result polling URLs.
    parts = endpoint.split("/", 2)
    app_id = f"{parts[0]}/{parts[1]}"

    status_url = f"https://queue.fal.run/{app_id}/requests/{request_id}/status"
    result_url = f"https://queue.fal.run/{app_id}/requests/{request_id}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        status_resp = requests.get(status_url, headers=headers, timeout=30)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get("status")

        if status == "COMPLETED":
            result_resp = requests.get(result_url, headers=headers, timeout=30)
            if not result_resp.ok:
                raise RuntimeError(
                    f"fal.ai result fetch failed ({result_resp.status_code}): "
                    f"{result_resp.text}"
                )
            return result_resp.json()
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"fal.ai request {status}: {status_data}"
            )

        time.sleep(2)

    raise TimeoutError(f"fal.ai request timed out after {timeout}s")


def _download_image_as_tensor(url):
    """Download an image URL and return a ComfyUI IMAGE tensor (B,H,W,C float32 0-1)."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _tensor_to_pil(tensor):
    """Convert a ComfyUI IMAGE tensor (B,H,W,C) to a PIL Image (first image in batch)."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_data_uri(img, fmt="png"):
    """Convert a PIL image to a data URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper() if fmt != "jpeg" else "JPEG")
    import base64
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = f"image/{fmt}" if fmt != "jpeg" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


class NanoBananaProTextToImage:
    """Generate images from text prompts using Nano Banana Pro via fal.ai."""

    ENDPOINT = "fal-ai/nano-banana-pro"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "fal_api_key": ("STRING", {"default": "", "multiline": False}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
                "output_format": (OUTPUT_FORMATS, {"default": "png"}),
                "resolution": (RESOLUTIONS, {"default": "1K"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "limit_generations": ("BOOLEAN", {"default": False}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 600}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate"
    CATEGORY = "fal.ai/Nano Banana Pro"

    def generate(
        self,
        prompt,
        fal_api_key="",
        num_images=1,
        aspect_ratio="1:1",
        output_format="png",
        resolution="1K",
        seed=-1,
        limit_generations=False,
        enable_web_search=False,
        timeout=300,
    ):
        api_key = _resolve_api_key(fal_api_key)

        payload = {
            "prompt": prompt,
            "num_images": 1 if limit_generations else num_images,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "resolution": resolution,
            "enable_web_search": enable_web_search,
        }
        if seed >= 0:
            payload["seed"] = seed

        result = _submit_request(self.ENDPOINT, payload, timeout, api_key)

        images_data = result.get("images", [])
        description = result.get("description", "")

        if not images_data:
            raise RuntimeError(
                f"No images returned. "
                f"Description: {description or 'none'}"
            )

        tensors = []
        for img_info in images_data:
            url = img_info.get("url", "")
            if url:
                tensors.append(_download_image_as_tensor(url))

        if not tensors:
            raise RuntimeError("No valid image URLs in response.")

        return (tensors, description or "")


class NanoBananaProImageEdit:
    """Edit images with natural language using Nano Banana Pro via fal.ai."""

    ENDPOINT = "fal-ai/nano-banana-pro/edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "fal_api_key": ("STRING", {"default": "", "multiline": False}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
                "output_format": (OUTPUT_FORMATS, {"default": "png"}),
                "resolution": (RESOLUTIONS, {"default": "1K"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "limit_generations": ("BOOLEAN", {"default": False}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "safety_tolerance": ("INT", {"default": 4, "min": 1, "max": 6}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 600}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "edit"
    CATEGORY = "fal.ai/Nano Banana Pro"

    def edit(
        self,
        image,
        prompt,
        fal_api_key="",
        image_2=None,
        image_3=None,
        num_images=1,
        aspect_ratio="1:1",
        output_format="png",
        resolution="1K",
        seed=-1,
        limit_generations=False,
        enable_web_search=False,
        safety_tolerance=4,
        timeout=300,
    ):
        api_key = _resolve_api_key(fal_api_key)

        # Convert primary image to data URI
        pil_img = _tensor_to_pil(image)
        image_url = _pil_to_data_uri(pil_img, output_format)

        payload = {
            "prompt": prompt,
            "image_url": image_url,
            "num_images": 1 if limit_generations else num_images,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "resolution": resolution,
            "enable_web_search": enable_web_search,
            "safety_tolerance": safety_tolerance,
        }
        if seed >= 0:
            payload["seed"] = seed

        # Add optional reference images
        if image_2 is not None:
            pil_2 = _tensor_to_pil(image_2)
            payload["image_2_url"] = _pil_to_data_uri(pil_2, output_format)

        if image_3 is not None:
            pil_3 = _tensor_to_pil(image_3)
            payload["image_3_url"] = _pil_to_data_uri(pil_3, output_format)

        result = _submit_request(self.ENDPOINT, payload, timeout, api_key)

        images_data = result.get("images", [])
        description = result.get("description", "")

        if not images_data:
            raise RuntimeError(
                f"No images returned. "
                f"Description: {description or 'none'}"
            )

        tensors = []
        for img_info in images_data:
            url = img_info.get("url", "")
            if url:
                tensors.append(_download_image_as_tensor(url))

        if not tensors:
            raise RuntimeError("No valid image URLs in response.")

        return (tensors, description or "")


NODE_CLASS_MAPPINGS = {
    "NanoBananaProTextToImage": NanoBananaProTextToImage,
    "NanoBananaProImageEdit": NanoBananaProImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProTextToImage": "Nano Banana Pro (Text to Image) 🍌",
    "NanoBananaProImageEdit": "Nano Banana Pro (Image Edit) 🍌",
}
