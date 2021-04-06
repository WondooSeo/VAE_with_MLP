import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

## Create Dataset

'''
! REMEMBER THE WAY FROM PNG TO DATASET !
PIL → NUMPY → TF.CONVERT_TO_TENSOR → TF.STACK → TF.DATASET
'''

path_dir = "C:/Users/mirac/Documents/PycharmProjects/VAE/Dataset_Thorax"
file_list = os.listdir(path_dir)
print(len(file_list))
temp_stacking = []

for img in file_list:
    np_img = np.asarray(Image.open("./Dataset/" + img))
    tensor_img = tf.convert_to_tensor(np_img)
    temp_stacking.append(tensor_img)

stack_img = tf.stack(temp_stacking) # Result → (Number of files, height, width, RGB)

dataset = tf.data.Dataset.from_tensor_slices(stack_img)
print(len(dataset))

class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train': self._generate_examples(path=extracted_path / 'train_images'),
        'test': self._generate_examples(path=extracted_path / 'test_images'),
    }

  def _generate_examples(self, path) -> Iterator[Tuple[Key, Example]]:
    """Generator of examples for each split."""
    for img_path in path.glob('*.jpeg'):
      # Yields (key, example)
      yield img_path.name, {
          'image': img_path,
          'label': 'yes' if img_path.name.startswith('yes_') else 'no',
      }