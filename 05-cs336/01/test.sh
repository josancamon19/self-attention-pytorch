# python src/train/transformer.py --tokens 2e8 --embedding-dim 768 --num-layers 6 --num-heads 12
python src/train/transformer.py --tokens 2e8 --embedding-dim 768 --num-layers 8 --num-heads 12
python src/train/transformer.py --tokens 2e8 --embedding-dim 768 --num-layers 12 --num-heads 12

python src/train/transformer.py --tokens 2e8 --embedding-dim 1024 --num-layers 6 --num-heads 8
# python src/train/transformer.py --tokens 2e8 --embedding-dim 1024 --num-layers 6 --num-heads 12 # assert % 2 == 0
python src/train/transformer.py --tokens 2e8 --embedding-dim 1024 --num-layers 6 --num-heads 16

python src/train/transformer.py --tokens 2e8 --embedding-dim 1024 --num-layers 8 --num-heads 16
python src/train/transformer.py --tokens 2e8 --embedding-dim 1024 --num-layers 12 --num-heads 16
python src/train/transformer.py --tokens 2e8 --embedding-dim 1024 --num-layers 16 --num-heads 16

# # can go higher, still at 50GB
python src/train/transformer.py --tokens 2e8 --embedding-dim 1024 --num-layers 20 --num-heads 16
python src/train/transformer.py --tokens 2e8 --embedding-dim 1024 --num-layers 24 --num-heads 16 # 71 GB

python src/train/transformer.py --tokens 2e8 --embedding-dim 1280 --num-layers 20 --num-heads 16 # 74 GB
python src/train/transformer.py --tokens 2e8 --embedding-dim 1280 --num-layers 20 --num-heads 20 # 78 GB
# # python src/train/transformer.py --tokens 2e8 --embedding-dim 1280 --num-layers 24 --num-heads 16 # no fit
