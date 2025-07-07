from helpers.plotters import plot_label_histogram
from helpers.renamer import rename_raw_images
from helpers.detector import detect_faces
from helpers.writer import write_wds_dataset
from trainers.default import train
from helpers.infer import infer_model
from helpers.labeler import relabel_images

# 1. Renomear imagens de acordo com labels
rename_raw_images('./data/raw', './data/renamed')

# 1.1. Histograma de classes
plot_label_histogram('data/renamed', './results/histograms')

# 2. Detectar faces com MTCNN
detect_faces('./data/renamed',
             './data/detected', color_threshold=None)

# 3. Reanotação de classes
relabel_images('./data/detected', './data/detected-skintones')

# 4. Escrita da base de dados
write_wds_dataset('./data/detected-skintones', './data/dataset-balanced-skintones/train.tar', './data/dataset-balanced-skintones/test.tar',
                  test_size=.2, max_count=None, balance=False, labels_to_ignore=[1, 2, 3, 4, 9, 10])

# 6. Treinamento
root_dir = '/content/gdrive/MyDrive/datamining'
# root_dir = './data'

train(
    epochs=100,
    data_dir=f'{root_dir}/dataset-full-skintones',
    dataset_type='webdataset',
    model_name='maxvit_nano_rw_256',
    batch_size=128,
    hyperparams={'lr': 1e-3, 'gamma': 0.95,
                 'step_size': 10, 'weight_decay': 1e-2},
    shuffle_size=57862,
    num_classes=4
)

# 7. Inferência
infer_model('./results/checkpoints')
