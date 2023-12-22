import argparse
import os
import zipfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--panorama-root',
        required=True,
        help='Folder with panorama archives',
        type=str)
    parser.add_argument(
        '--output-panorama-root',
        required=True,
        help='Folder with unziped panoramas',
        type=str)
    parser.add_argument(
        '--bb-root',
        required=True,
        help='Folder with 3d bounding boxes and annotations',
        type=str)
    parser.add_argument(
        '--output-bb-root',
        required=True,
        help='Folder for unziped 3d bounding boxes and annotations',
        type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_panorama_root):
        os.mkdir(args.output_panorama_root)

    if not os.path.exists(args.output_bb_root):
        os.mkdir(args.output_bb_root)

    sorted_panorama_root = sorted(
        os.listdir(args.panorama_root),
        key=lambda x: int(x[:-4].split('_')[-1]))
    for idx, zip in enumerate(sorted_panorama_root):
        print(idx, zip)
        out_path = os.path.join(args.output_panorama_root, str(idx))
        with zipfile.ZipFile(
            os.path.join(args.panorama_root, zip), 'r') as z:
            for name in z.namelist():
                try:
                    z.extract(name, out_path)
                except Exception as e:
                    print(e)

        print(f'File - {zip} is successfully unziped to the {out_path} folder')

    with zipfile.ZipFile(os.path.join(
        args.bb_root, os.listdir(args.bb_root)[0])) as z:
        z.extractall(args.output_bb_root)
        print(
            f'File - {os.listdir(args.bb_root)[0]} is successfully '
            f'unziped to the {args.output_bb_root} folder')
