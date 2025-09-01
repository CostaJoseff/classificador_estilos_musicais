from sklearn.model_selection import train_test_split
import os, shutil, random

dir_base = "musicas"
dir_h5 = "h5"

estilos = os.listdir(dir_base)
for estilo in estilos:
    h5_path = os.path.join(dir_base, estilo, dir_h5)
    specs = os.listdir(h5_path)
    random.shuffle(specs)
    indice_70 = int(len(specs) * 0.7)
    train = specs[:indice_70]
    test = specs[indice_70:]


    train_path = os.path.join(h5_path, "train")
    test_path = os.path.join(h5_path, "test")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for spec in train:
        spec_path = os.path.join(h5_path, spec)
        shutil.move(spec_path, train_path)

    for spec in test:
        spec_path = os.path.join(h5_path, spec)
        shutil.move(spec_path, test_path)
        