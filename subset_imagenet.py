import os, json, shutil
os.system("rm -rf imagenet_1k")

in_path = "imagenet/ILSVRC/ImageSets/CLS-LOC/val.txt"
save_dir = "imagenet_1k"
os.makedirs(save_dir, exist_ok=1)

N = 100
with open(in_path, "r") as f:
    ls = f.readlines()
    ls = ls[:N]

save_path = os.path.join(save_dir, "val_map.txt")
with open(save_path, "w+") as f:
    for l in ls:
        f.write(l)
print(save_path)
    

in_path = "imagenet/ILSVRC/ImageSets/CLS-LOC/train_cls.txt"

N = 1000
with open(in_path, "r") as f:
    ls = f.readlines()
    ls = ls[:N]

save_path = os.path.join(save_dir, "train_map.txt")
with open(save_path, "w+") as f:
    for l in ls:
        f.write(l)
print(save_path)

# ---------------------------------------------------------------------------------------------
tr_dir = os.path.join("imagenet_1k", "train")
os.makedirs(tr_dir,exist_ok=1)
with open(save_path, 'r') as f:
    ls = f.readlines()
    for l in ls:
        ll = l.split(" ")
        fnames = ll[0].split("_")
        folder = fnames[0].split("/")[0]
        img_path = os.path.join("imagenet/ILSVRC/Data/CLS-LOC/train", f"{ll[0]}.JPEG")
        save_dir = os.path.join(tr_dir, folder)
        os.makedirs(save_dir, exist_ok=1)

        save_path = os.path.join(save_dir, f"{ll[0].split('/')[1]}.JPEG")
        shutil.copyfile(img_path, save_path)
        print("saved:", save_path)

val_dir = os.path.join("imagenet_1k", "val")
os.makedirs(val_dir,exist_ok=1)
with open("imagenet_1k/val_map.txt", 'r') as f:
    ls = f.readlines()
    for l in ls:
        ll = l.split(" ")
        fname = ll[0]
        img_path = os.path.join("imagenet/ILSVRC/Data/CLS-LOC/val", f"{fname}.JPEG")
        save_path = os.path.join(val_dir, f"{fname}.JPEG")
        shutil.copyfile(img_path, save_path)
        print("saved:", save_path)

print("Done")
