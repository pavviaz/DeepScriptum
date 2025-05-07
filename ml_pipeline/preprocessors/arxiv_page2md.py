import os
import random
import base64
import io
import re
import html
from typing import List, Dict, Any, Optional, Tuple

import orjson
from PIL import Image, ImageOps
from tqdm import tqdm


COORD_PRECISION = 3


def decode_image_from_base64(base64_string: str) -> Optional[Image.Image]:
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception:
        return None


def extract_text_from_xhtml(xhtml_string: str) -> str:
    text = re.sub(r"<[^>]+>", " ", xhtml_string)
    text = html.unescape(text)
    text = " ".join(text.split()).lower()
    return text


def get_non_latex_words(markdown_string: str) -> List[str]:
    words = re.findall(r"(?<!\\)\b[a-zA-Z]{2,}\b", markdown_string)
    return [word.lower() for word in words]


def resize_and_pad_image(
    img: Image.Image, target_w: int, target_h: int
) -> Tuple[Image.Image, float, float, int, int]:
    orig_w, orig_h = img.size

    # Compute scale to fit within target_w x target_h while maintaining aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize the image
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Compute padding (centered)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    # Pad the image with black borders
    img_padded = ImageOps.expand(
        img_resized, (pad_left, pad_top, pad_right, pad_bottom), fill=0
    )

    # Compute scale factors based on actual resized dimensions
    scale_w = new_w / orig_w
    scale_h = new_h / orig_h

    return img_padded, scale_w, scale_h, pad_left, pad_top


def get_coords_string(
    coords: List[int],
    scale_w: float,
    scale_h: float,
    pad_left: int,
    pad_top: int,
    target_width: int,
    target_height: int,
    normalize: bool,
    log_obj: Any,
    img_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not coords or len(coords) != 4:
        if img_info:
            log_obj.info(f"Invalid coords list: {coords} for image info {img_info}")
        else:
            log_obj.info(f"Invalid coords list: {coords}")
        return None

    l_orig, t_orig, r_orig, b_orig = coords

    # Transform coordinates based on resizing and padding
    l_trans = l_orig * scale_w + pad_left
    t_trans = t_orig * scale_h + pad_top
    r_trans = r_orig * scale_w + pad_left
    b_trans = b_orig * scale_h + pad_top

    if normalize:
        if target_width <= 0 or target_height <= 0:
            log_obj.info(
                f"Invalid target dimensions for normalization: {target_width}x{target_height}"
            )
            return None
        norm_l = max(0.0, min(1.0, l_trans / target_width))
        norm_t = max(0.0, min(1.0, t_trans / target_height))
        norm_r = max(0.0, min(1.0, r_trans / target_width))
        norm_b = max(0.0, min(1.0, b_trans / target_height))

        if norm_l >= norm_r:
            norm_r = norm_l + 1e-6
        if norm_t >= norm_b:
            norm_b = norm_t + 1e-6
        norm_r = min(1.0, norm_r)
        norm_b = min(1.0, norm_b)

        coord_str = (
            "["
            + f"{norm_l:.{COORD_PRECISION}f}, "
            + f"{norm_t:.{COORD_PRECISION}f}, "
            + f"{norm_r:.{COORD_PRECISION}f}, "
            + f"{norm_b:.{COORD_PRECISION}f}"
            + "]"
        )
    else:
        l_trans = int(max(0, min(target_width - 1, l_trans)))
        t_trans = int(max(0, min(target_height - 1, t_trans)))
        r_trans = int(max(l_trans + 1, min(target_width, r_trans)))
        b_trans = int(max(t_trans + 1, min(target_height, b_trans)))
        coord_str = f"[{l_trans}, {t_trans}, {r_trans}, {b_trans}]"

    return coord_str


def process_markdown_images(
    markdown_string: str,
    on_page_images: Dict[str, Dict[str, Any]],
    current_page_index: int,
    scale_w: float,
    scale_h: float,
    pad_left: int,
    pad_top: int,
    target_img_width: int,
    target_img_height: int,
    normalize_coords_flag: bool,
    log_obj: Any,
) -> Optional[str]:
    processed_markdown = markdown_string
    image_pattern = re.compile(
        r"!\[.*?\]\(([^)]+\.(?:png|jpg|jpeg|gif|bmp|svg))\)", re.IGNORECASE
    )
    matches = list(image_pattern.finditer(markdown_string))

    mentioned_image_pages = []
    replacements = {}

    for match in matches:
        image_filename = match.group(1)
        placeholder_full = match.group(0)

        if image_filename not in on_page_images:
            log_obj.info(
                f"Image '{image_filename}' in MD on page {current_page_index} not found in metadata. Skipping page."
            )
            return None

        img_info = on_page_images[image_filename]
        image_page = img_info.get("page")
        coords = img_info.get("coords")

        if image_page is None:
            log_obj.info(
                f"Image '{image_filename}' metadata lacks 'page' info. Skipping page."
            )
            return None
        image_page -= 1  # Adjust to match index

        mentioned_image_pages.append(image_page)

        if image_page != current_page_index:
            log_obj.info(
                f"Image '{image_filename}' mentioned on page {current_page_index} belongs to page {image_page}. Skipping page."
            )
            return None

        if not coords or len(coords) != 4:
            log_obj.info(
                f"Missing or invalid coords for image '{image_filename}' on page {current_page_index}. Coords: {coords}. Skipping page."
            )
            return None

        coord_str = get_coords_string(
            coords=coords,
            scale_w=scale_w,
            scale_h=scale_h,
            pad_left=pad_left,
            pad_top=pad_top,
            target_width=target_img_width,
            target_height=target_img_height,
            normalize=normalize_coords_flag,
            log_obj=log_obj,
            img_info=img_info,
        )

        if coord_str is None:
            log_obj.info(
                f"Failed to get coordinate string for image '{image_filename}' on page {current_page_index}. Skipping page."
            )
            return None

        if placeholder_full not in replacements:
            replacements[placeholder_full] = f"![image]({coord_str})"

    if len(mentioned_image_pages) > 1:
        is_monotonic = all(
            mentioned_image_pages[i] <= mentioned_image_pages[i + 1]
            for i in range(len(mentioned_image_pages) - 1)
        )
        if not is_monotonic:
            log_obj.info(
                f"Page indices of images mentioned in markdown on page {current_page_index} are not monotonic: {mentioned_image_pages}. Skipping page."
            )
            return None

    for match in matches:
        placeholder_full = match.group(0)
        if placeholder_full in replacements:
            replacement_text = replacements[placeholder_full]
            processed_markdown = processed_markdown.replace(
                placeholder_full, replacement_text
            )

    return processed_markdown


def preprocessor(
    path: str,
    log_obj: Any,
    task: Any,
    amount: Optional[int] = None,
    markdown_text_similarity_threshold: float = 0.6,
    target_image_size: Optional[Tuple[int, int]] = None,
    normalize_img_coords: bool = True,
) -> List[Tuple[Image.Image, str]]:
    processed_data = []
    original_count = 0
    discarded_count = 0
    processed_item_count = 0

    if not os.path.exists(path):
        error_msg = f"Data file '{path}' doesn't exist"
        log_obj.invoke_exception(error_msg, OSError, task)
        return []

    log_obj.info(f"Starting preprocessing for file: {path}")
    try:
        with open(path, "rb") as f:
            json_data = f.read()
            _data = orjson.loads(json_data)
        original_count = len(_data)
        log_obj.info(f"Loaded {original_count} raw data points.")
    except Exception as e:
        error_msg = f"Failed to load or parse JSON from '{path}': {e}"
        log_obj.invoke_exception(error_msg, type(e), task)
        return []

    if amount and amount < original_count:
        log_obj.info(f"Sampling {amount} data points randomly.")
        _data = random.sample(_data, amount)
        original_count = amount

    for item in tqdm(_data):
        processed_item_count += 1
        if processed_item_count % 100 == 0:
            log_obj.info(
                f"Attempted processing {processed_item_count}/{len(_data)} items..."
            )

        doi = item.get("doi", "unknown_doi")
        page_index = item.get("page_index", -1)
        llm_markdown = item.get("llm_markdown")
        xhtml_text = item.get("xhtml_text")
        on_page_images = item.get("on_page_images", {})
        page_screenshot_b64 = item.get("page_screenshot")

        item_id = f"{doi}_{page_index}"

        if not all([llm_markdown, xhtml_text, page_screenshot_b64, page_index != -1]):
            log_obj.info(f"Skipping item {item_id}: Missing essential field(s).")
            discarded_count += 1
            continue

        pil_image = decode_image_from_base64(page_screenshot_b64)
        if pil_image is None:
            log_obj.info(f"Skipping item {item_id}: Failed to decode page screenshot.")
            discarded_count += 1
            continue
        original_width, original_height = pil_image.size

        # Handle image resizing and padding
        if target_image_size:
            target_h, target_w = target_image_size  # (height, width)
            try:
                final_pil_image, scale_w, scale_h, pad_left, pad_top = (
                    resize_and_pad_image(pil_image, target_w, target_h)
                )
                target_w, target_h = target_w, target_h  # Use target dimensions
            except Exception as e:
                log_obj.info(
                    f"Skipping item {item_id}: Failed to resize and pad image to {target_image_size}. Error: {e}"
                )
                discarded_count += 1
                continue
        else:
            final_pil_image = pil_image
            scale_w, scale_h = 1.0, 1.0
            pad_left, pad_top = 0, 0
            target_w, target_h = original_width, original_height

        xhtml_plain_text = extract_text_from_xhtml(xhtml_text)
        markdown_words = get_non_latex_words(llm_markdown)

        if not xhtml_plain_text and markdown_words:
            log_obj.info(
                f"Skipping item {item_id}: XHTML text is empty but Markdown is not."
            )
            discarded_count += 1
            continue

        if markdown_words and xhtml_plain_text:
            found_words = sum(1 for word in markdown_words if word in xhtml_plain_text)
            similarity_ratio = found_words / len(markdown_words)
            if similarity_ratio < markdown_text_similarity_threshold:
                log_obj.info(f"Skipping item {item_id}: Low text similarity")
                discarded_count += 1
                continue
        elif not markdown_words:
            pass

        processed_markdown = process_markdown_images(
            llm_markdown,
            on_page_images,
            page_index,
            scale_w,
            scale_h,
            pad_left,
            pad_top,
            target_w,
            target_h,
            normalize_img_coords,
            log_obj,
        )

        if processed_markdown is None:
            discarded_count += 1
            continue

        processed_data.append((final_pil_image, processed_markdown))

    final_count = len(processed_data)
    log_obj.info(f"Preprocessing finished. Kept {final_count} items.")
    if original_count > 0:
        actual_discard_count = original_count - final_count
        discard_ratio = actual_discard_count / original_count
        log_obj.info(f"Discarded {actual_discard_count} items ({discard_ratio:.2%}).")
    else:
        log_obj.info("Warning: Original data count was zero.")

    if not processed_data:
        log_obj.info("CRITICAL WARNING: No data points remaining after preprocessing!")

    return processed_data


if __name__ == "__main__":
    class MockLogger:
        def info(self, text):
            print(text)

        def invoke_exception(self, text, *args):
            print(text)

    preprocessor(
        "/Users/pavelvyaznikov/Downloads/arxiv_dataset_exp250k-llm-1k.json",
        MockLogger(),
        None,
        markdown_text_similarity_threshold=0.95,
        target_image_size=(892, 768),
    )
