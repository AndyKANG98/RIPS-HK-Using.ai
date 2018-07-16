
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
protoc object_detection/protos/*.proto --python_out=.

models = ['ssdlite_mobilenet_v2_coco_2018_05_09', 'faster_rcnn_resnet50_coco_2018_01_28', 
          'faster_rcnn_inception_v2_coco_2018_01_28', 'ssd_mobilenet_v2_coco_2018_03_29', 
          'ssd_inception_v2_coco_2018_01_28']

model_name='ssdlite_mobilenet_v2_coco_2018_05_09'

# train
pipeline_config_path="pipeline_config_files/$model_name".config
train_path="train/$model_name"
python object_detection/train.py \
  --logtostderr \
  --pipeline_config_path=$pipeline_config_path \
  --train_dir=$train_path

# eval job
checkpoint_path="train/%s/model.ckpt-10589" % model_name
pipeline_config_path="pipeline_config_files/$model_name".config
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=$pipeline_config_path\
    --checkpoint_dir=$checkpoint_path/ \
    --eval_dir=results/eval


# extract the frozen inference graph
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# paths
pipeline_config_path="pipeline_config_files/$model_name.config"
checkpoint_path="train/$model_name/model.ckpt-20000"
output_path="frozen_graph/$model_name"

python object_detection/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=$pipeline_config_path \
--trained_checkpoint_prefix=$checkpoint_path \
--output_directory=$output_path


# Run on test set
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

input_path="./test_images/img"
output_path="./results/img"
label_path="./annotations/label_map.pbtxt"
frozen_graph_path="./frozen_graph/$model_name/frozen_inference_graph.pb"

# run on test set
python object_detection/inference.py \
  --input_dir=$input_path \
  --output_dir=$output_path \
  --label_map=$label_path \
  --frozen_graph=$frozen_graph_path \
  --num_output_classes=1

