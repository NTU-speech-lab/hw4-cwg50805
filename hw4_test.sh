wget https://github.com/NTU-speech-lab/hw4-cwg50805/releases/download/Model/ckpt_best.model
wget https://github.com/NTU-speech-lab/hw4-cwg50805/releases/download/Model/w2v_all.model
wget https://github.com/NTU-speech-lab/hw4-cwg50805/releases/download/Model/w2v_all.model.trainables.syn1neg.npy
wget https://github.com/NTU-speech-lab/hw4-cwg50805/releases/download/Model/w2v_all.model.wv.vectors.npy

python3 main.py 1 $1 $2

