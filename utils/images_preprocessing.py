import os
from PIL import Image, ImageOps
import cv2
import albumentations as alb
import numpy as np
from tqdm import tqdm

RESIZE_W = 601
RESIZE_H = 85
PATH = "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\formula_images_png_4_resized\\"

test_transform = alb.Compose(  # sharp formulas
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
    ]
)


def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.BILINEAR)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')


def crop_image(img, output_path):
    """Crops image to content
    Args:
        img: (string) path to image
        output_path: (string) path to output image
    """
    old_im = Image.open(img).convert('L')
    old_im.show()
    img_data = np.asarray(old_im, dtype=np.uint8) # height, width
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        old_im.save(output_path)
        return False

    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
    #old_im.save(output_path)
    old_im.show()
    #return True


def pad_image(img, output_path, pad_size=[8,8,8,8], buckets=None):
    """Pads image with pad size and with buckets
    Args:
        img: (string) path to image
        output_path: (string) path to output image
        pad_size: list of 4 ints
        buckets: ascending ordered list of sizes, [(width, height), ...]
    """
    top, left, bottom, right = pad_size
    old_im = Image.open(img)
    old_size = (old_im.size[0] + left + right, old_im.size[1] + top + bottom)
    new_size = get_new_size(old_size, buckets)
    new_im = Image.new("RGB", new_size, (255,255,255))
    new_im.paste(old_im, (left, top))
    new_im.show()


def get_new_size(old_size, buckets):
    """Computes new size from buckets
    Args:
        old_size: (width, height)
        buckets: list of sizes
    Returns:
        new_size: original size or first bucket in iter order that matches the
            size.
    """
    if buckets is None:
        return old_size
    else:
        w, h = old_size
        for (w_b, h_b) in buckets:
            if w_b >= w and h_b >= h:
                return w_b, h_b

        return old_size


def Reformat_Image(ImageFilePath, name):
    '''
    Pads image to square size
    '''
    image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if(width != height):
        bigside = width if width > height else height

        background = Image.new('RGBA', (bigside, bigside), (255, 255, 255, 255))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))

        background.paste(image, offset)
        background = background.resize((650, 650), Image.BILINEAR)
        background.save(f"C:/Users/shace/Documents/GitHub/MachineLearning/PythonML/formula_images_png_3_squared/{name}")
    #     print("Image has been resized !")

    # else:
    #     print("Image is already a square, it has not been resized !")


def visualize(image, name):
    cv2.imshow(name,image)
    cv2.waitKey(0)


def alb_test(image_path:str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualize(image, "before_alb")

    augmented_image = test_transform(image=image)['image']
    visualize(augmented_image, "after_alb")


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return ImageOps.invert(Image.fromarray(tensor))


def save_alb_images(path): # image quality drops after saving
    dir = os.listdir(path)
    for d in tqdm(dir):
        image = cv2.imread(path + d)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = test_transform(image=image)['image']
        #plt.imsave("C:\\Users\\shace\\Desktop\\temp_2\\" + d, augmented_image)
        #aug_img = tf.cast(augmented_image/255.0, tf.float32)
        #tensor_to_image(aug_img).save("C:\\Users\\shace\\Desktop\\temp_2\\" + d)
        #im.save("C:\\Users\\shace\\Desktop\\temp_2\\" + d)
        cv2.imwrite("C:\\Users\\shace\\Desktop\\temp_2\\" + d, augmented_image)


def width_height_info(path):
    '''
    Get some info about images in directory
    '''
    dir = os.listdir(path)
    w = []
    h = []
    w_m = 0
    h_m = 0
    for d in tqdm(dir):
        img = Image.open(path + d)
        w.append(img.size[0])
        h.append(img.size[1])
        w_m += img.size[0]
        h_m += img.size[1]
    print(f"w_m = {w_m} ; h_m = {h_m}\nw_m = {w_m / len(dir)} ; h_m = {h_m / len(dir)}\nw_max = {max(w)} ; h_max = {max(h)}")


def make_fix_size(path):
    '''
    Paste image to bigger fixed size blank image and pad it
    '''
    dir = os.listdir(path)
    for d in tqdm(dir):
        img = Image.open(path + d)
        if img.size[0] <= RESIZE_W and img.size[1] <= RESIZE_H:
            background = Image.new('RGBA', (RESIZE_W, RESIZE_H), (255, 255, 255, 255))
            background.paste(img, (0, 0))
            background.save(f"C:\\Users\\shace\\Desktop\\formula_images_png_4_resized\\{d}")
        else:
            print(f"image {d} is too big! w = {img.size[0]} ; h = {img.size[1]}")


