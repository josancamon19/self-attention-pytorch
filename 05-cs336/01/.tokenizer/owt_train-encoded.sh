cd .tokenizer/
# wget "https://storage.googleapis.com/test-joan1/owt_train-encoded.npy.gz"
curl -C - -o owt_train-encoded.npy.gz "https://storage.googleapis.com/test-joan1/owt_train-encoded.npy.gz"
gunzip owt_train-encoded.npy.gz
cd ..