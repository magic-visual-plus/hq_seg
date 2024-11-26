import sys
import os
import shutil


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    filenames = os.listdir(input_path)
    for filename in filenames:
        if filename.endswith('_mask.png'):
            # it is mask file
            continue
        else:
            img_path = os.path.join(input_path, filename)
            img_id = os.path.splitext(filename)[0]
            mask_path = os.path.join(input_path, img_id + '_watershed_mask.png')

            if not os.path.exists(mask_path):
                continue

            print(f'copy img_id:{img_id}')
            output_img_path = os.path.join(output_path, filename)
            output_mask_path = os.path.join(output_path, img_id + '_mask.png')
            shutil.copy(img_path, output_img_path)
            shutil.copy(mask_path, output_mask_path)
            pass
        pass
    pass