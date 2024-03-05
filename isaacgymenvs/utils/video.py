import os
import os.path as osp
import imageio.v2 as imageio
import argparse
import yaml
import cv2

from tqdm import tqdm

def img_to_vid():
    """Probably broken"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ext", type=str, choices=["mp4", "gif"], default="gif")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    in_folder = osp.join("recorded_frames", args.exp_name, "images")
    out_folder = osp.join("recorded_frames", args.exp_name, "videos")
    assert osp.isdir(in_folder), f"Input folder {in_folder} does not exist"
    os.makedirs(out_folder, exist_ok=True)

    assert osp.exists(osp.join(in_folder, "../metadata.yml"))
    metadata = yaml.load(open(osp.join(in_folder, "../metadata.yml"), "r"), Loader=yaml.FullLoader)
    num_envs = metadata["num_envs"]

    fnames = sorted(os.listdir(in_folder))
    images = [[] for _ in range(num_envs)]
    for fname in fnames:
        basename, ext = osp.splitext(fname)
        if ext != ".png":
            continue
        env_id = int(basename.split("-")[-1])
        try:
            images[env_id].append(imageio.imread(osp.join(in_folder, fname)))
        except OSError:
            print(f"Failed to read {fname}")
        except Exception as e:
            print(e)
            
    if args.ext == "gif":
        kw = {"duration": 1000 // args.fps}
    else:
        kw = {"fps": args.fps}
        
    for env_id in tqdm(range(num_envs)):
        if images[env_id]:
            imageio.mimsave(
                osp.join(out_folder, f"video_{env_id}.{args.ext}"), 
                images[env_id], 
                format=args.ext.upper(),
                **kw
            )

def vid_to_img():
    def extract_frames(video_path, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = osp.join(output_folder, f"frame{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        cap.release()
        print(f"{frame_count} frames extracted from {video_path}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    assert osp.exists(args.input)
    if osp.isfile(args.input):
        par_dir = osp.dirname(args.input)
        filenames = [osp.basename(args.input)]
    elif osp.isdir(args.input):
        par_dir = args.input
        filenames = [x for x in os.listdir(par_dir) if osp.splitext(x)[1] == ".mp4"]
    for filename in filenames:
        extract_frames(osp.join(par_dir, filename), osp.join(par_dir, osp.splitext(filename)[0]))


if __name__ == "__main__":
    vid_to_img()