
SCRIPTS_ROOT=/home/eustinova/scripts/VisionLabsReportMay2016/

########## VIPER
MODEL_FILE=${SCRIPTS_ROOT}/models/train_val_model_v2_gpu.prototxt

FILESET=/${SCRIPTS_ROOT}/descriptors/viper/viperfiles.txt
WEIGHTS_FILE=${SCRIPTS_ROOT}/models/viper/train_iter_3000.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/viper/descriptors_v2_gpu.txt
srun --gres=gpu:1 -w hpc2  ./.build_debug/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}


MODEL_FILE=${SCRIPTS_ROOT}/models/train_val_model_v2_cpu.prototxt

FILESET=/${SCRIPTS_ROOT}/descriptors/viper/viperfiles.txt
WEIGHTS_FILE=${SCRIPTS_ROOT}/models/viper/train_iter_3000.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/viper/descriptors_v2_cpu.txt
./.build_debug/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} cpu ${FILESET} ${OUT}




########### CUHK03

FILESET=${SCRIPTS_ROOT}/descriptors/cuhk03/cuhk03files.txt
MODEL_FILE=${SCRIPTS_ROOT}/models/train_val_model_v2_gpu.prototxt

WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/labeled_split1.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/labeled_split1_v2_gpu.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}
