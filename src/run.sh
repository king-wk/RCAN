#########################################################################################
# train
# train_dataset DIV2K

# 正常模型
# 从已训练的模型继续训练
# for scale in 2
# do
#     LOG=./../experiment/RCAN_G10R20P48B16_X${scale}-`date +%Y-%m-%d-%H-%M-%S`.txt
#     echo -e "python main.py --model RCAN --save RCAN_G10R20P48B16_X${scale} --load RCAN_G10R20P48B16_X${scale} --scale ${scale} --n_resgroups 10 --n_resblocks 20 --n_feats 64  --pre_train ./../experiment/RCAN_G10R20P48B16_X${scale}/model/model_lastest.pt --save_results --save_models --valid --patch_size 48 2>&1 | tee ${LOG}"
#     python main.py --model RCAN --save RCAN_G10R20P48B16_X${scale} --load RCAN_G10R20P48B16_X${scale} --scale ${scale} --n_resgroups 10 --n_resblocks 20 --n_feats 64  --pre_train ./../experiment/RCAN_G10R20P48B16_X${scale}/model/model_lastest.pt --save_results --save_models --valid --patch_size 48 2>&1 | tee ${LOG}
# done

# 重新训练
# for scale in 4
# do
#     LOG=./../experiment/RCAN_G10R20P48B16_X${scale}-`date +%Y-%m-%d-%H-%M-%S`.txt
#     echo -e "python main.py --model RCAN --save RCAN_G10R20P48B16_X${scale} --scale ${scale} --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --save_results --save_models --valid --patch_size 48 2>&1 | tee ${LOG}"
#     python main.py --model RCAN --save RCAN_G10R20P48B16_X${scale} --scale ${scale} --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --save_results --save_models --valid --patch_size 48 2>&1 | tee ${LOG}
# done

# 会随机跳出的模型

# # 从已训练的模型继续训练
# for scale in 4
# do
#     LOG=./../experiment/RCAN_RO_G10R20P48B16_X${scale}-`date +%Y-%m-%d-%H-%M-%S`.txt
#     echo -e "python main.py --model RCAN --save RCAN_RO_G10R20P48B16_X${scale} --load RCAN_RO_G10R20P48B16_X${scale} --scale ${scale} --n_resgroups 10 --n_resblocks 20 --n_feats 64  --pre_train ./../experiment/RCAN_RO_G10R20P48B16_X${scale}/model/model_lastest.pt --save_results --save_models --random_output --valid --patch_size 48 2>&1 | tee ${LOG}"
#     python main.py --model RCAN --save RCAN_RO_G10R20P48B16_X${scale} --load RCAN_RO_G10R20P48B16_X${scale} --scale ${scale} --n_resgroups 10 --n_resblocks 20 --n_feats 64  --pre_train ./../experiment/RCAN_RO_G10R20P48B16_X${scale}/model/model_lastest.pt --save_results --save_models --random_output --valid --patch_size 48 2>&1 | tee ${LOG}
# done

# 重新训练
for scale in 3
do
    LOG=./../experiment/RCAN_RO_G10R20P48B16_X${scale}-`date +%Y-%m-%d-%H-%M-%S`.txt
    echo -e "python main.py --model RCAN --save RCAN_RO_G10R20P48B16_X${scale} --scale ${scale} --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --save_results --save_models --random_output --valid --patch_size 48 2>&1 | tee ${LOG}"
    python main.py --model RCAN --save RCAN_RO_G10R20P48B16_X${scale} --scale ${scale} --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --save_results --save_models --random_output --valid --patch_size 48 2>&1 | tee ${LOG}
done

#########################################################################################
# test
# scale 2, 3, 4
# test_dataset DIV2K Set4 Set14 B100 Urban100

# # 正常模型
# for scale in 2 3 4
# do
#     for dataset in DIV2K Set4 Set14 B100 Urban100
#     do
#         echo -e "python main.py --data_test ${dataset} --save Test_RCAN_G10R20P48B16_X${scale} --scale ${scale} --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ./../experiment/RCAN_G10R20P48B16_X${scale}/model/model_lastest.pt --test_only --save_results"
#         python main.py --data_test ${dataset} --save Test_RCAN_G10R20P48B16_X${scale} --scale ${scale} --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ./../experiment/RCAN_G10R20P48B16_X${scale}/model/model_lastest.pt --test_only --save_results
#     done
# done

# # 会随机跳出的模型
# for scale in 2 3 4
# do
#     for dataset in DIV2K Set4 Set14 B100 Urban100
#     do
#         echo -e "python main.py --data_test ${dataset} --save Test_RCAN_RO_G10R20P48B16_X${scale} --scale ${scale} --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ./../experiment/RCAN_RO_G10R20P48B16_X${scale}/model/model_lastest.pt --test_only --save_results"
#         python main.py --data_test ${dataset} --save Test_RCAN_RO_G10R20P48B16_X${scale} --scale ${scale} --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ./../experiment/RCAN_RO_G10R20P48B16_X${scale}/model/model_lastest.pt --test_only --save_results
#     done
# done