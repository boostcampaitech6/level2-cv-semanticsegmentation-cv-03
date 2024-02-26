import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def rle_to_mask(rle, height, width):
    s = np.array(rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths

    mask = np.zeros(height * width, dtype=np.int32)
    mask[starts] += 1
    mask[ends] -= 1

    mask = np.cumsum(mask)
    return mask.reshape(height, width).astype(np.uint8)


def mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def main():
    csv_files = [
        '/data/ephemeral/home/output.csv',
        '/data/ephemeral/home/output-2.csv',
    ]

    weights = [0.7, 0.3]

    combined_df = [pd.read_csv(f) for f in csv_files]

    class_name = combined_df[0]['class'].tolist()
    assert len(class_name) == 8352

    image_class = []
    rles = []

    height, width = 2048, 2048

    for i in tqdm(range(len(class_name))):
        rles_mask = []
        
        for df, w in zip(combined_df, weights):
            weighted_masks = []
            if type(df.iloc[i]['rle']) == float:
                weighted_masks.append(np.zeros((height, width)))
                continue
            weighted_masks.append(rle_to_mask(df.iloc[i]['rle'], height, width) * w)
        
        combined_mask = sum(weighted_masks)

        combined_mask[combined_mask < sum(weights) / 2] = 0
        combined_mask[combined_mask >= sum(weights) / 2] = 1
        combined_mask = combined_mask.astype(np.uint8)

        image = np.zeros((height, width), dtype=np.uint8)
        image += combined_mask
        
        rles.append(mask_to_rle(image))
        image_class.append(f"{combined_df[0].iloc[i]['image_name']}_{combined_df[0].iloc[i]['class']}")


    filename, classes = zip(*[x.split("_") for x in image_class])
    image_name = [os.path.basename(f) for f in filename]

    submission = pd.DataFrame({
                    "image_name": image_name,
                    "class": classes,
                    "rle": rles,
                })

    save_dir = '/data/ephemeral/home/save_dir'
    submission.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)


if __name__ == '__main__':
    main()
