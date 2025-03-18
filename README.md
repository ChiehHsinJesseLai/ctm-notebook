## CTM Training and Evaluation
This is the training repository of ctm using lsun bedroom dataset
 
 
## Prerequisites 
1.Install docker. 
2.create and launch container. The build.sh and launch.sh scripts are given in docker folder
 
## Dataset
Use LSUN bedroom dataset. Download the lsun dataset from official site. Then arrange the data using LSUN_Bedroom_Download/read_write_lsun.py 
 
 ## Training
For CTM+DSM training, run commands/CTM+DSM_command.sh
Recommendation: at least run CTM+DSM for ~50k iterations

For CTM+DSM+GAN training, run commands/CTM+DSM+GAN_command.sh
Recommendation: at least run CTM+DSM+GAN for >=30k iterations
