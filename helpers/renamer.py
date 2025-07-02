import os
import pandas as pd


def get_year_and_state_from_csv(csv_path: str):
    file_name, _ = csv_path.rsplit('.', 1)
    _, _, year, state = file_name.split('_')

    return year, state


def rename_raw_images(input_path: str, output_path: str):
    if not os.path.exists(f'{output_path}'):
        os.makedirs(f'{output_path}')

    files = list(filter(lambda x: ".csv" in x, os.listdir(input_path)))

    for file_name in files:
        df = pd.read_csv(f'{input_path}/{file_name}', delimiter=';',
                         quotechar='"', encoding='latin')
        year, state = get_year_and_state_from_csv(file_name)
        print(year, state, file_name)

        for _, row in df.iterrows():
            label = row['CD_COR_RACA']
            img_id = row['SQ_CANDIDATO']
            image_name = f'F{state}{img_id}_div'
            image_paths = list(
                filter(lambda x: image_name in x, os.listdir(input_path)))

            if (img_id != '-4' and label != '-4' and int(label) < 6 and len(image_paths) > 0):
                image_path = image_paths[0]
                _, extension = image_path.split('.')
                new_image_name = f'{img_id}_{year}_{state}_{label}.{extension}'

                print(new_image_name)

                os.system(
                    f'cp {input_path}/{image_path} {output_path}/{new_image_name}')
