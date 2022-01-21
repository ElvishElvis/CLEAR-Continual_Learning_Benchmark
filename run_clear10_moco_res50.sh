CUDA_VISIBLE_DEVICES=0 python3 train.py \
--data_folder_path '/data3/zhiqiul/yfcc_dynamic_10/dynamic_300/images' \
--class_list 'NEGATIVE baseball bus camera cosplay dress hockey laptop racing soccer sweater' \
--method 'LwF GDumb BiasReservoir_Fixed_reset_0.2 BiasReservoir_Fixed_reset_1.0 BiasReservoir_Fixed_reset_0.5
BiasReservoir_Fixed_reset_2.0 BiasReservoir_Fixed_reset_5.0 
BiasReservoir_Dynamic_1.0 BiasReservoir_Dynamic_0.75 BiasReservoir_Dynamic_reset_1.0 BiasReservoir_Dynamic_reset_0.75 BiasReservoir_Dynamic_reset_0.5 
BiasReservoir_Dynamic_reset_0.25 BiasReservoir_Dynamic_0.5 
BiasReservoir_Dynamic_0.25 BiasReservoir_Fixed_0.2 BiasReservoir_Fixed_0.5 BiasReservoir_Fixed_1.0 
BiasReservoir_Fixed_2.0 BiasReservoir_Fixed_5.0 
Cumulative CWRStar JointTraining EWC SynapticIntelligence Replay AGEMFixed Reservoir
GDumbFinetune'  \
--split  'clear10_moco_res50' \
--restart '0' \
--nepoch 100 \
--step_schedular_decay 60 \
--schedular_step 0.1 \
--batch_size 8 \
--start_lr 1 \
--weight_decay 0. \
--momentum  0.9 \
--timestamp  10 \
--num_classes  11 \
--num_instance_each_class 300 \
--random_seed 1111 \
--test_split 0.3 \
--feature_path '/data/jiashi/' \
--pretrain_feature 'moco_resnet50_clear_10_feature' 
