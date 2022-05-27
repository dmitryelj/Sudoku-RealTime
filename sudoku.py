# Sudoku real-time solving and recognition
# Dmitrii <dmitryelj@gmail.com>

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont
from typing import List, Any
from datetime import datetime
import imageio
import time
import numpy as np
import pytesseract
import solver
import random
import copy
import sys
import glob
import logging


# Basic params
model_file_name = "ocr_model.pt"
mnist_folder = 'mnist'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Use ESC to close app window
ESC_KEY = 27

# Images size for OCR
IMG_SIZE = 32


# Neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        kernel_size = 5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=0)
        out_layer_size = ((IMG_SIZE - kernel_size + 1) - kernel_size + 1)//2
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*out_layer_size*out_layer_size, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Digits Dataset Generator
class DigitsDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.fonts = glob.glob("fonts/*.ttf")
        self.fonts_dict = {}
        self.digits_ = [None] * self.__len__()
        self.generate_all()

    def __len__(self):
        return 60000

    def __getitem__(self, index):
        return self.digits_[index]

    def generate_all(self):
        logging.info("Generating the digits dataset...")
        t_start = time.monotonic()
        for p in range(self.__len__()):
            if p % 10000 == 0:
                logging.debug(f"  {p} of {self.__len__()}...")
            self.digits_[p] = self.generate_digit()
        logging.info(f"Done, dT={time.monotonic() - t_start}s\n")

    def generate_digit(self):
        digit = random.randint(0, 9)
        data = self.generate_digit_pil(digit)
        return data, digit

    def generate_digit_pil(self, digit: int):
        text = str(digit)
        area_size = 2*IMG_SIZE
        img = Image.new("L", (area_size, area_size), (0,))
        draw = ImageDraw.Draw(img)
        font_name, font_size = random.choice(self.fonts), random.randint(48, 64)
        font_key = f"{font_name}-{font_size}"
        if font_key not in self.fonts_dict:
            self.fonts_dict[font_key] = ImageFont.truetype(font_name, font_size)
        font = self.fonts_dict[font_key]
        text_x = area_size//2 + random.randint(-2, 2)
        text_y = area_size//2 - random.randint(-1, 1)
        draw.text((text_x, text_y), text, (255,), font=font, anchor="mm")
        transform = transforms.Compose([transforms.Resize([IMG_SIZE, IMG_SIZE]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        resized = transform(img)[0].unsqueeze(0)
        return resized


def ocr_show_dataset(dataset):
    images = []
    for r in range(10):
        hor_images = []
        for d in range(10):
            digit_img = cv2.copyMakeBorder(dataset[10*r + d][0].reshape(IMG_SIZE, IMG_SIZE).detach().numpy(),
                                           2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(128,))
            hor_images.append(digit_img)
        hor_line = np.concatenate(hor_images, axis=1)
        images.append(hor_line)
    cv2.imshow("Dataset", np.concatenate(images, axis=0))
    while True:
        if cv2.waitKey(1) & 0xFF == ESC_KEY:
            break
        time.sleep(0.1)


def ocr_train():
    max_epochs = 12
    batch_size_train = 10
    log_interval = 600
    # MNIST dataset loader
    # dataset = datasets.MNIST(mnist_folder, train=True, download=True,
    #                          transform=transforms.Compose([transforms.Resize([IMG_SIZE, IMG_SIZE]),
    #                                                        transforms.ToTensor(),
    #                                                        transforms.Normalize((0.1307,), (0.3081,))]))
    # TTF-rendered digits dataset
    dataset = DigitsDataset()
    # ocr_show_dataset(dataset)
    # return
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

    # Create model
    model = Model().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)

    # Train
    model.train()
    logging.info("Training the model, use CUDA: %s", use_cuda)
    for epoch in range(max_epochs):
        logging.debug(f"Epoch: {epoch + 1} of {max_epochs}")
        t1 = time.monotonic()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                logging.debug('  Train [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(batch_idx * len(data), len(train_loader.dataset),
                                                                              100 * batch_idx / len(train_loader), loss.item()))
            total_loss += loss.item()
        logging.debug(f"  Total loss: {total_loss}, dT={time.monotonic() - t1}s")

    torch.save(model.state_dict(), model_file_name)
    logging.info("Model saved to %s", model_file_name)
    return model


def ocr_load_model(device):
    model = Model()
    model.load_state_dict(torch.load(model_file_name, map_location=torch.device(device)))
    model.eval()
    return model


def predict_tesseract(images: List):
    results = []
    for x, y, img_x, img_y, digit_img in images:
        value = predict_digit_tesseract(digit_img, x, y)
        results.append(value)
    return results


def predict_digit_tesseract(digit_img: Any, x: int, y: int):
    w, h = digit_img.shape
    if w > h:  # Convert image to square size
        digit_img = cv2.copyMakeBorder(digit_img, 0, 0, (w - h)//2, w - h - (w - h)//2, cv2.BORDER_CONSTANT, value=(255,))
    digit_img = cv2.copyMakeBorder(digit_img, w//10, w//10, w//10, w//10, cv2.BORDER_CONSTANT, value=(255,))
    # cv2.imwrite(f"digit_{x}x{y}.png", digit_img)  # Save digits as files for debugging
    # Run OCR
    res = pytesseract.image_to_string(digit_img, config='-l eng --psm 8 --dpi 70 -c tessedit_char_whitelist=0123456789').strip()
    return int(res[0:1]) if len(res) > 0 else None


def predict_pytorch(model: Model, images: List):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize([IMG_SIZE, IMG_SIZE]),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    # Prepare images for the recognition
    images_ = []
    for x, y, img_x, img_y, digit_img in images:
        w, h = digit_img.shape
        # Convert image to square size
        if w > h:
            img_square = cv2.copyMakeBorder(digit_img, 10, 10, 10 + (w - h)//2, 10 + w - h - (w - h)//2, cv2.BORDER_CONSTANT, value=(255,))
        else:
            img_square = cv2.copyMakeBorder(digit_img, 10 + (h - w)//2, 10 + h - w - (h - w)//2, 10, 10, cv2.BORDER_CONSTANT, value=(255,))
        # cv2.imwrite(f"digit_{x}x{y}.png", img_square)  # For debugging only
        data = transform(~img_square).unsqueeze(0)
        images_.append(data)
    # Convert separated images to the single Pytorch tensor
    if len(images_) == 0:
        return []
    data = torch.cat(images_)
    # Run OCR model
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        pred = output.data.max(1, keepdim=True)[1].reshape((len(images_),))
        return pred.tolist()


def process_image(model: Model, img: Any, show_preview=True):
    w, h = img.shape[1], img.shape[0]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(img_gray, 100, 400)

    blurred = cv2.GaussianBlur(img_gray, (11, 11), 0)
    img_bw = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # cv2.imwrite("img_out.png", img_out)

    t_start = time.monotonic()

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_out = img.copy()
    # cv2.drawContours(img_out, contours, -1, (0, 255, 0), 1)
    result_found = False
    for cntr in contours:
        imgx, imgy, imgw, imgh = cv2.boundingRect(cntr)
        if imgw < w/5 or imgw < h/5 or imgw/imgh < 0.25 or imgw/imgh > 1.5:
            continue

        def normalize_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        # Approximate the contour and apply the perspective transform
        peri = cv2.arcLength(cntr, True)
        frm = cv2.approxPolyDP(cntr, 0.1*peri, True)
        if len(frm) != 4:
            continue

        # Converted image should fit into the original size
        board_size = max(imgw, imgh)
        if len(frm) != 4 or imgx + board_size >= w or imgy + board_size >= h:
            continue
        # Points should not be too close to each other (use euclidian distance)
        if cv2.norm(frm[0][0] - frm[1][0], cv2.NORM_L2) < 0.1*peri or \
           cv2.norm(frm[2][0] - frm[1][0], cv2.NORM_L2) < 0.1*peri or \
           cv2.norm(frm[3][0] - frm[1][0], cv2.NORM_L2) < 0.1*peri or \
           cv2.norm(frm[3][0] - frm[2][0], cv2.NORM_L2) < 0.1*peri:
            continue

        # Draw sudoku contour
        cv2.line(img_out, frm[0][0], frm[1][0], (0, 200, 0), thickness=3)
        cv2.line(img_out, frm[1][0], frm[2][0], (0, 200, 0), thickness=3)
        cv2.line(img_out, frm[2][0], frm[3][0], (0, 200, 0), thickness=3)
        cv2.line(img_out, frm[0][0], frm[3][0], (0, 200, 0), thickness=3)
        cv2.drawContours(img_out, frm, -1, (0, 255, 255), 10)

        # Source and destination points for the perspective transform
        src_pts = normalize_points(frm.reshape((4, 2)))
        dst_pts = np.array([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]], dtype=np.float32)
        t_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        _, t_matrix_inv = cv2.invert(t_matrix)

        # Convert images, colored and monochrome
        warped_disp = cv2.warpPerspective(img, t_matrix, (board_size, board_size))
        warped_bw = cv2.warpPerspective(img_bw, t_matrix, (board_size, board_size))

        # Sudoku board found, extract digits from the 9x9 grid
        images = []
        cell_w, cell_h = board_size//9, board_size//9
        for x in range(9):
            for y in range(9):
                x1, y1, x2, y2 = x*cell_w, y*cell_h, (x + 1)*cell_w, (y + 1)*cell_h
                cx, cy, w2, h2 = (x1 + x2)//2, (y1 + y2)//2, cell_w, cell_h
                # Find the contour of the digit
                crop = warped_bw[y1:y2, x1:x2]
                cntrs, _ = cv2.findContours(crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for dc in cntrs:
                    imgx2, imgy2, imgw2, imgh2 = cv2.boundingRect(dc)
                    if 0.2 * w2 < imgw2  < 0.8 * w2 and 0.4 * h2 < imgh2 < 0.8 * h2:
                        cv2.rectangle(warped_disp, (x1 + imgx2, y1 + imgy2), (x1 + imgx2 + imgw2, y1 + imgy2 + imgh2), (0, 255, 0), 1)
                        digit_img = crop[imgy2:imgy2 + imgh2, imgx2:imgx2 + imgw2]
                        images.append((x, y, cx, cy, digit_img))
                        break

        # Draw image at the top corner of display
        if show_preview:
            disp_size = h//3
            disp_x, disp_y = w - disp_size - 10, 10
            warp_disp_resized = cv2.resize(warped_disp, (disp_size, disp_size), interpolation=cv2.INTER_AREA)
            img_out[disp_y:disp_y + disp_size, disp_x:disp_x + disp_size] = warp_disp_resized

        # t_start = time.monotonic()
        # Recognise extracted digits
        results = predict_pytorch(model, images)  # results = predict_tesseract(images)

        # Draw OCR results
        board = [0]*(9*9)
        for (x, y, img_x, img_y, digit_img), result in zip(images, results):
            if result:
                board[9*x + y] = result

                # Calculate coordinates on the original image
                pt_orig = cv2.perspectiveTransform(np.array([[[img_x, img_y]]], dtype=np.float32), t_matrix_inv).reshape((2,)).astype(np.int32)
                cv2.putText(img_out, str(result), pt_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2, cv2.LINE_AA, False)

        # print("dT1:", time.monotonic() - t_start)
        # print("Board:", board)

        # Solve the board and draw digits on the board
        t_start = time.monotonic()
        board_orig = list(board)
        res = solver.solve_c(board)
        # print("dT2:", time.monotonic() - t_start)
        if res:
            result_found = True
            for x in range(9):
                for y in range(9):
                    if board_orig[9*x + y] == 0:
                        pt_x, pt_y = x*cell_w + cell_w//2, y*cell_h + cell_h//2
                        pt_orig = cv2.perspectiveTransform(np.array([[[pt_x, pt_y]]], dtype=np.float32), t_matrix_inv).reshape((2,)).astype(np.int32)
                        cv2.putText(img_out, str(board[9*x + y]), pt_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA, False)
                        pass

    # Additional info
    cv2.putText(img_out, "Solution Found" if result_found else "Solution Not Found", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA, False)
    if show_preview:
        dt = time.monotonic() - t_start
        cv2.putText(img_out, f"FPS: {1/dt:.2f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA, False)
    return result_found, img_out


def process_webcam(model):
    cap = cv2.VideoCapture(0)
    width, height, fps = 1280, 720, 15
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    logging.info(f"Starting the video stream: {width}x{height}")

    mp4_writer, mp4_filename = None, ""
    gif_writer, gif_filename = None, ""
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_orig = frame.copy()
        try:
            # Process
            res, img_out = process_image(model, frame)
            # Add frame for saving if active
            if mp4_writer is not None:
                # gif_frames.append(img_out.copy())
                mp4_writer.write(img_out)
                cv2.putText(img_out, "Saving %s..." % mp4_filename, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA, False)
            if gif_writer is not None:
                gif_writer.append_data(img_out)
                cv2.putText(img_out, "Saving %s..." % gif_filename, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA, False)
            # Display
            cv2.imshow('Frame', img_out)
        except Exception as e:
            logging.error("Exception: %s, image saved to 'crash.png'", e)
            cv2.imwrite("crash.png", frame_orig)
            break

        # Process key codes
        key_code = cv2.waitKey(1)
        if key_code & 0xFF == ESC_KEY:
            break
        if key_code & 0xFF == ord('f'):
            # Save single frame
            filename = datetime.now().strftime("Frame-%Y%m%d%H%M%S.png")
            cv2.imwrite(filename, frame_orig)
            logging.info("File %s saved" % filename)
        if key_code & 0xFF == ord('g'):
            # Save frames stream as GIF
            if gif_writer is None:
                gif_filename = datetime.now().strftime("Video-%Y%m%d%H%M%S.gif")
                gif_writer = imageio.get_writer(gif_filename, mode='I')
                logging.info("File %s will be saved" % gif_filename)
            else:
                gif_writer.close()
                gif_writer = None
                logging.info("Done")
        if key_code & 0xFF == ord('v'):
            # Save frames stream as MP4
            if mp4_writer is None:
                mp4_filename = datetime.now().strftime("Video-%Y%m%d%H%M%S.mp4")
                logging.info("Video recording %s started" % mp4_filename)
                mp4_writer = cv2.VideoWriter(mp4_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            else:
                mp4_writer.release()
                mp4_writer = None
                logging.info("Done")

    cap.release()
    if mp4_writer:
        mp4_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    logging.info("Sudoku Real-time Decoder 1.0\n")
    logging.info("Usage:\npython3 sudoku.py --train\tTrain the OCR model\npython3 sudoku.py sudoku.png\tDecode local file\npython3 sudoku.py\t\tOpen the web camera stream\n")
    logging.info("Hotkeys:")
    logging.info("Esc - exit, f - save current frame, v - start/stop video recording, g - start/stop gif recording\n")

    # Train the OCR Model
    if len(sys.argv) > 1 and '--train' in sys.argv[1]:
        ocr_train()
        exit(0)

    # Load OCR Model
    device = "cpu"
    model = ocr_load_model(device).to(device)

    # Analyse the local file
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        _, img_out = process_image(model, img, show_preview=False)
        logging.info("Done")
        cv2.imshow('Image', img_out)
        # Wait for Esc to exit
        while True:
            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()
        exit(0)

    # Analyse the webcam stream
    try:
        process_webcam(model)
    except KeyboardInterrupt:
        pass
