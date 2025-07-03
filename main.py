from helpers.plotters import plot_label_histogram, plot_hsv_histogram
# from helpers.renamer import rename_raw_images
from helpers.detector import detect_faces
from helpers.pca import get_pca_from_dir
from helpers.writer import write_wds_dataset
from trainers.default import train

# 1. Renomear imagens de acordo com labels
# rename_raw_images('./data/raw', './data/renamed')

# 3. Histograma de classes
# plot_label_histogram('data/renamed', './results/histograms')

# Como não há amostras significativas para as classes 4 e 5, elas serão ignoradas

# 2. Detectar faces com MTCNN
# detect_faces('./data/renamed',
#              './data/detected', color_threshold=None)


# 4. Visualização com PCA (Conjunto reduzido)
# get_pca_from_dir('./data/detected-test', './results/pca', n_components=3)

# 5. Escrita da base de dados
# write_wds_dataset('./data/detected-test', './data/dataset-full-roi/train.tar', './data/dataset-full-roi/test.tar',
#                   test_size=.2, max_count=None, balance=False, labels_to_ignore=[4, 5])

# 5.1. Metadados da base de dados

# Balanced Dataset
# {1: 10029, 3: 10029, 2: 10029} totals
# 23974 train_count
# 6113 test_count

# Full Dataset
# {1: 32700, 3: 29515, 2: 10029} totals
# 57735 train_count
# 14508 test_count


# 6. Treinamento
root_dir = '/content/gdrive/MyDrive/datamining'
# root_dir = './data'
train(200, f'{root_dir}/dataset-full-roi',
      'webdataset', 'maxvit_nano_rw_256', batch_size=64, hyperparams={'lr': 1e-3, 'gamma': 0.95, 'step_size': 10, 'weight_decay': 1e-2}, shuffle_size=23974)

# 7. Visualização do treinamento com PCA
