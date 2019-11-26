from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import os


checkpoint_path = os.path.join(os.getcwd(), "checkpoints/pnasnet-5_large_model.ckpt")
# print_tensors_in_checkpoint_file(file_name=checkpoint_path,all_tensors=False, tensor_name='')

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
items = []
items.append(reader.debug_string().decode("utf-8"))


with open('./pnasnet_exclude_scopes.txt', 'w') as f:
    for item in items:
        f.write("%s\n" % item)
        print(item)
