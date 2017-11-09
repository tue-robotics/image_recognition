import cv2
import os
import fnmatch


def read_annotated(dir_path, patterns=["*.jpg", "*.png", "*.jpeg"], include_path=False):
    """
    Read annotated images from a directory. This reader assumes that the images in this directory are separated in
    different directories with the label name as directory name. The method returns a generator of the label (string)
    and the opencv image.
    :param dir_path: The base directory we are going to write read
    :param patterns: Patterns of the images the reader should match
    """
    for label in os.listdir(dir_path):
        for root, dirs, files in os.walk(os.path.join(dir_path, label)):
            for basename in files:
                for pattern in patterns:
                    if fnmatch.fnmatch(basename, pattern):
                        fullpath = os.path.join(root, basename)
                        if not include_path:
                            yield label, cv2.imread(fullpath)
                        else:
                            yield label, cv2.imread(fullpath), fullpath

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Get image classifications via some TensorFlow network')

    parser.add_argument('-a', '--annotated-dir', type=str, help='List the images in th given directory',
                    required=False,
                    default=".")

    args = parser.parse_args()

    for label, image, path in read_annotated(args.annotated_dir, include_path=True):
        print(label, path)