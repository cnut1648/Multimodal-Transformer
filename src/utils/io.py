from PIL import Image
import numpy as np
import os

def load_img(path):
	return Image.open(path).convert('RGB')


def sample_frames(frame_dir, num_frames=8):
	frame_list = sorted(os.listdir(frame_dir))

	return [frame_list[i] for i in np.linspace(0, len(frame_list)-1, num_frames).astype('int')]

