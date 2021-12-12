python3 /path/to/train_v2/onepass/P3_input.py --level 1 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1
wait

python3 /path/to/train_v2/onepass/train_P3.py --level 1 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 200 --num_neuron 16 
wait

python3 /path/to/train_v2/onepass/P1_P2_input.py --level 1 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait

python3 /path/to/train_v2/onepass/train_P1.py --level 1 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 300 --num_neuron 16
wait

python3 /path/to/train_v2/onepass/train_P2.py --level 1 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 300 --num_neuron 16
wait

python3 /path/to/train_v2/onepass/U_input.py --level 1 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/train_U.py --level 1 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 200 --num_neuron 16




python3 /path/to/train_v2/onepass/approx.py --level 1 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait





python3 /path/to/train_v2/onepass/P3_input.py --level 2 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait

python3 /path/to/train_v2/onepass/train_P3.py --level 2 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 300 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/P1_P2_input.py --level 2 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/train_P1.py --level 2 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 300 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/train_P2.py --level 2 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 300 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/U_input.py --level 2 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/train_U.py --level 2 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 200 --num_neuron 16





python3 /path/to/train_v2/onepass/approx.py --level 2 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait




python3 /path/to/train_v2/onepass/P3_input.py --level 3 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait

python3 /path/to/train_v2/onepass/train_P3.py --level 3 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 300 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/P1_P2_input.py --level 3 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/train_P1.py --level 3 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 300 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/train_P2.py --level 3 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 300 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/U_input.py --level 3 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1 --num_neuron 16
wait
python3 /path/to/train_v2/onepass/train_U.py --level 3 --method 'onepass_42_AG_16n' --transform '42_AG' --dynamic 1  --epochs 200 --num_neuron 16

