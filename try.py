from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import pandas as pd
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def get_dali_pipeline(crop_size):
    
    data_dir = "/home/ICT2000/jxu/Multimodal-Transformer/data/data/NTU RGB+D/nturgb+d_rgb"
    csv_dir = "/home/ICT2000/jxu/Multimodal-Transformer/data/datasets/NTU RGB+D"
    df = pd.read_csv(f"{csv_dir}/post_val.csv")
    files, labels = (data_dir + "/" + df["path"]).tolist(), df["label"].tolist()

    images = fn.readers.video(
        device="gpu", filenames=files, random_shuffle=False, sequence_length=8,
        name="Reader", dtype=types.UINT8, pad_last_batch=True, image_type=types.RGB,)
    images = fn.crop(images, crop=crop_size, dtype=types.FLOAT,
                     crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
                     crop_pos_y=fn.random.uniform(range=(0.0, 1.0)))
    images = fn.transpose(images, perm=[3, 0, 1, 2])

    
    return images

pipeline = get_dali_pipeline(
                    num_threads=1,
                    batch_size=4,
                    crop_size=224)
pipeline.build()
train_data = DALIGenericIterator(
    pipeline, ["data"],
   reader_name='Reader', auto_reset=True,
)


for i, data in enumerate(train_data):
  x = data[0]['data']
#   pred = model(x)
#   loss = loss_func(pred, y)
#   backward(loss, model)