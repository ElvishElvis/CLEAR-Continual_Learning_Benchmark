CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py \
--data_folder_path '/data3/zhiqiul/clear_datasets/CLEAR100-V0/labeled_images' \
--data_train_path '' \
--data_test_path '' \
--class_list 'airplane amusement_park aquarium athletics bag bar baseball basketball beer bicycle 
boat bookstore boot bread bridge bus_station cable cake camera car castle chair cheese 
chef chocolate church classroom coat cosplay denim dress esport farm ferry firefighter 
fountain gallery glasses glove golf guitar hat headphone helicopter highway hockey 
horse_riding ice_cream keyboard lamppost laptop makeup microphone mobile_phone mug 
museum necklace noodle piano pizza police projector pump railway restaurant ring 
roller_skating salad sandal sandwich scarf shopping_mall skateboarding skating 
skyscraper snowboarding sock sofa soldier sport_shoes stadium statue subway 
supermarket sushi sweatshirt swimming swimsuit table television temple tennis 
tie toilet toy vase violin volleyball wine zoo' \
--method 'BiasReservoir_Fixed_reset_0.2 BiasReservoir_Fixed_reset_0.5 BiasReservoir_Fixed_reset_1.0 
BiasReservoir_Fixed_reset_2.0 BiasReservoir_Fixed_reset_5.0 
BiasReservoir_Dynamic_1.0 BiasReservoir_Dynamic_0.75 BiasReservoir_Dynamic_reset_1.0 BiasReservoir_Dynamic_reset_0.75 BiasReservoir_Dynamic_reset_0.5 
BiasReservoir_Dynamic_reset_0.25 BiasReservoir_Dynamic_0.5 
BiasReservoir_Dynamic_0.25 BiasReservoir_Fixed_0.2 BiasReservoir_Fixed_0.5 BiasReservoir_Fixed_1.0 
BiasReservoir_Fixed_2.0 BiasReservoir_Fixed_5.0 
Cumulative CWRStar JointTraining EWC SynapticIntelligence Replay AGEMFixed Reservoir'  \
--split  'clear100_moco_res50' \
--restart '0' \
--nepoch 100 \
--step_schedular_decay 60 \
--schedular_step 0.1 \
--batch_size 16 \
--start_lr 1 \
--weight_decay 0. \
--momentum  0.9 \
--timestamp  10 \
--num_classes  100 \
--num_instance_each_class 1000 \
--num_instance_each_class_test 1000 \
--random_seed 1111 \
--test_split 0.3 \
--feature_path '/data/jiashi/' \
--pretrain_feature 'moco_resnet50_clear_100_feature' \
--max_memory_size 3000
