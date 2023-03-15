# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os.path
import cv2
import argparse
from tqdm import tqdm
import cupy as cp

def create_noise_mask(h, w):
    # 创建mask
    mask = cp.zeros((h, w), dtype=bool)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    mask = cp.bitwise_not(mask)
    return mask


def add_random_noise(fft_frame, mask):
    noise = cp.random.normal(0, 0.2, fft_frame.shape)
    # 根据mask添加噪声
    fft_frame[mask] += noise[mask]
    return fft_frame


def process_frame(frame, frame_counter, noise_mask):
    channels = cv2.split(frame)
    result_channels = []

    for channel in channels:
        channel_gpu = cp.array(channel)
        f_channel = cp.fft.fft2(channel_gpu)
        fshift_channel = cp.fft.fftshift(f_channel)

        fshift_channel = add_random_noise(fshift_channel, noise_mask)

        f_ishift_channel = cp.fft.ifftshift(fshift_channel)
        if_channel = cp.fft.ifft2(f_ishift_channel)
        if_channel = cp.real(if_channel)
        if_channel = cp.clip(if_channel, 0, 255)  # 限制数值范围
        if_channel = cp.asnumpy(if_channel.astype(cp.uint8))  # 修改数据类型
        result_channels.append(if_channel)

    return cv2.merge(result_channels)


def frame_generator(cap):
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, frame_counter
        frame_counter += 1


def main(video_file, output_file):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if os.path.exists(output_file):
        os.remove(output_file)

    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=True)

    noise_mask = create_noise_mask(height, width)

    with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
        for frame, frame_counter in frame_generator(cap):
            processed_frame = process_frame(frame, frame_counter, noise_mask)
            out.write(processed_frame)
            pbar.update(1)

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add random noise to video frames in the frequency domain.")
    parser.add_argument("-i", "--input", required=True, help="Path to input video file.")
    parser.add_argument("-o", "--output", required=True, help="Path to output video file.")
    args = parser.parse_args()

    main(args.input, args.output)
