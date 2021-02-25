import argparse
import os
import pathlib

exts = ('.cpp', '.h', '.cu', '.txt', '.in')
exclude = ('3rdparty', 'doc', '.git', '.vs', 'build_win')


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename, 'r') as f:
        s = f.read()
        if old_string not in s:
            print('"{}" not found in {}'.format(old_string.rstrip("\n"), filename))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        # s = f.read()
        print('Changing "{}" to "{}" in {}'.format(old_string.rstrip("\n"), new_string.rstrip("\n"), filename))
        s = s.replace(old_string, new_string)
        f.write(s)


def scan_folder(root, old_string, new_string):
    for path, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in exclude]
        for name in files:
            if name.endswith(exts):
                file_path = pathlib.PurePath(path, name)
                inplace_change(file_path, old_string, new_string)


# python prepare_release.py old_version new_version
# python prepare_release.py 0.3.0 0.3.1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replacing versions ECVL tool')
    parser.add_argument('old_version', type=str, help='The old version number')
    parser.add_argument('new_version', type=str, help='The new version number')
    args = parser.parse_args()
    old_version, new_version = args.old_version, args.new_version
    old_string = f' Version: {old_version}\n'
    new_string = f' Version: {new_version}\n'

    scan_folder(pathlib.Path('.').absolute(), old_string, new_string)
