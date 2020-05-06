wget https://github.com/NTU-speech-lab/hw4-cwg50805/releases/download/Model/ckpt_best.model
wget https://github.com/NTU-speech-lab/hw4-cwg50805/releases/download/Modeal_new/w2v_test.model
wget https://github.com/NTU-speech-lab/hw4-cwg50805/releases/download/Modeal_new/w2v_test.model.trainables.syn1neg.npy
wget https://github.com/NTU-speech-lab/hw4-cwg50805/releases/download/Modeal_new/w2v_test.model.wv.vectors.npy

python3 main.py 1 $1 $2

