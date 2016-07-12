

SCRIPTS_ROOT=/home/eustinova/scripts/VisionLabsReportMay2016/

MODEL_FILE=${SCRIPTS_ROOT}/models/train_val_model.prototxt




########### VIPER

FILESET=/${SCRIPTS_ROOT}/descriptors/viper/viperfiles.txt
WEIGHTS_FILE=models/viper/train_iter_3000.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/viper/descriptors.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}






##############CUHK03
FILESET=${SCRIPTS_ROOT}/descriptors/cuhk03/cuhk03files.txt




WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/labeled_split1.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/labeled_split1.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}
 
WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/labeled_split2.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/labeled_split2.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}

WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/labeled_split3.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/labeled_split3.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}

WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/labeled_split4.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/labeled_split4.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}

WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/labeled_split5.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/labeled_split5.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}



WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/detected_split1.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/detected_split1.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}
 
WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/detected_split2.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/detected_split2.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}

WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/detected_split3.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/detected_split3.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}

WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/detected_split4.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/detected_split4.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}

WEIGHTS_FILE=${SCRIPTS_ROOT}/models/cuhk03/detected_split5.caffemodel
OUT=${SCRIPTS_ROOT}/descriptors/cuhk03/detected_split5.txt
srun --gres=gpu:1 -w hpc2  ./.build_release/examples/re-id/reid_descriptor_prediction.bin  ${MODEL_FILE} ${WEIGHTS_FILE} gpu ${FILESET} ${OUT}



