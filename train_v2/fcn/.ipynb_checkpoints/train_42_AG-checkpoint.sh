#CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/P3_input.py --level 1 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
#wait

#CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P3.py --level 1 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'
#wait

#CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/P1_P2_input.py #--level 1 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
#wait

#CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P1.py --level 1 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16  --Loss 'round_mse'& CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P2.py --level 1 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'
#wait

#CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/U_input.py --level 1 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16
#wait
#CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_U.py --level 1 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'




CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/approx.py --level 1 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
wait





CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/P3_input.py --level 2 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P3.py --level 2 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/P1_P2_input.py --level 2 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P1.py --level 2 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse' &
CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P2.py --level 2 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/U_input.py --level 2 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_U.py --level 2 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'






CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/approx.py --level 2 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
wait





CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/P3_input.py --level 3 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P3.py --level 3 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/P1_P2_input.py --level 3 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P1.py --level 3 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse' & CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_P2.py --level 3 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/U_input.py --level 3 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --num_neuron 16 
wait

CUDA_VISIBLE_DEVICES=0 python3 /data/tdardour_data/image_comp/codes/train_v2/fcn/train_U.py --level 3 --method 'fcn_round_42_AG_16n' --transform '42_AG' --dynamic 0 --epochs 200  --num_neuron 16 --Loss 'round_mse'
