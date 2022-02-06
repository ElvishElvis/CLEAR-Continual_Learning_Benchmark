CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../train.py \
--data_folder_path '/data3/zhiqiul/clear_datasets/CLEAR50-V2/labeled_images' \
--class_list 'baseball basketball billard boats boot bowling bridge camera car castle chair 
church coat dress ferry football gallery glasses glove golf guitar hat headphone highway 
hockey lamppost laptop lego microphone mobile_phone mug museum opera_theatre projector 
pump railway sandal shopping_mall soccer sock sofa subway supermarket swimsuit television 
temple tennis traffic_light truck vase violin volleyball' \
--method 'LwF GDumb BiasReservoir_Fixed_reset_0.2 BiasReservoir_Fixed_reset_0.5 BiasReservoir_Fixed_reset_1.0 
BiasReservoir_Fixed_reset_2.0 BiasReservoir_Fixed_reset_5.0 
BiasReservoir_Dynamic_1.0 BiasReservoir_Dynamic_0.75 BiasReservoir_Dynamic_reset_1.0 BiasReservoir_Dynamic_reset_0.75 BiasReservoir_Dynamic_reset_0.5 
BiasReservoir_Dynamic_reset_0.25 BiasReservoir_Dynamic_0.5 
BiasReservoir_Dynamic_0.25 BiasReservoir_Fixed_0.2 BiasReservoir_Fixed_0.5 BiasReservoir_Fixed_1.0 
BiasReservoir_Fixed_2.0 BiasReservoir_Fixed_5.0 
Cumulative CWRStar JointTraining EWC SynapticIntelligence Replay AGEMFixed Reservoir'  \
--split  'clear50_moco_res50' \
--restart '0' \
--nepoch 100 \
--step_schedular_decay 60 \
--schedular_step 0.1 \
--batch_size 16 \
--start_lr 1 \
--weight_decay 0. \
--momentum  0.9 \
--timestamp  10 \
--num_classes  52 \
--num_instance_each_class 1000 \
--random_seed 1111 \
--test_split 0.3 \
--feature_path '/data/jiashi/' \
--pretrain_feature 'moco_resnet50_clear_50_feature' 
--max_memory_size 3000
