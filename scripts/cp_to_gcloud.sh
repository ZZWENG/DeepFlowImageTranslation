#!/usr/bin/env bash


instance_name='zhenzhenweng@instance-gpu2'

subfolders='data modules real_nvp_module scripts'

for name in $subfolders
do
    gcloud compute scp --recurse $name $instance_name:/home/zhenzhenweng/myproject
done


files='train.py'
for name in $files
do
    gcloud compute scp $name $instance_name:/home/zhenzhenweng/myproject
done

# tar -zcvf datasets.tar.gz datasets/
#gcloud compute scp datasets.tar.gz $instance_name:/home/zhenzhenweng/myproject
# tar xvzf datasets.tar.gz