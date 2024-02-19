for dataset in BlogCatalog ACM YelpChi weibo reddit Cora questions tolokers
do
    mkdir ../logs/NoisyCLF_log/${dataset}
done

# Pretrain diffusion models on all datasets
echo "training diffusion models"
python ../src/train_graphdiffusion_uncond.py --loss_type l1 --dataset weibo
python ../src/train_graphdiffusion_uncond.py --loss_type l2 --dataset weibo
python ../src/train_graphdiffusion_uncond.py --loss_type huber --dataset weibo

python ../src/train_graphdiffusion_uncond.py --loss_type l1 --dataset reddit
python ../src/train_graphdiffusion_uncond.py --loss_type l2 --dataset reddit
python ../src/train_graphdiffusion_uncond.py --loss_type huber --dataset reddit

python ../src/train_graphdiffusion_uncond.py --loss_type l1 --dataset YelpChi
python ../src/train_graphdiffusion_uncond.py --loss_type l2 --dataset YelpChi
python ../src/train_graphdiffusion_uncond.py --loss_type huber --dataset YelpChi

python ../src/train_graphdiffusion_uncond.py --loss_type l1 --dataset tolokers
python ../src/train_graphdiffusion_uncond.py --loss_type l2 --dataset tolokers
python ../src/train_graphdiffusion_uncond.py --loss_type huber --dataset tolokers

python ../src/train_graphdiffusion_uncond.py --loss_type l1 --dataset questions
python ../src/train_graphdiffusion_uncond.py --loss_type l2 --dataset questions
python ../src/train_graphdiffusion_uncond.py --loss_type huber --dataset questions

python ../src/train_graphdiffusion_uncond.py --loss_type l1 --dataset Cora
python ../src/train_graphdiffusion_uncond.py --loss_type l2 --dataset Cora
python ../src/train_graphdiffusion_uncond.py --loss_type huber --dataset Cora

python ../src/train_graphdiffusion_uncond.py --loss_type l1 --dataset BlogCatalog
python ../src/train_graphdiffusion_uncond.py --loss_type l2 --dataset BlogCatalog
python ../src/train_graphdiffusion_uncond.py --loss_type huber --dataset BlogCatalog

python ../src/train_graphdiffusion_uncond.py --loss_type l1 --dataset ACM
python ../src/train_graphdiffusion_uncond.py --loss_type l2 --dataset ACM
python ../src/train_graphdiffusion_uncond.py --loss_type huber --dataset ACM

# Pretrian NoisyCLF on all datasets
python ../src/train_classifier.py  --dataset reddit
python ../src/train_classifier.py  --dataset weibo
python ../src/train_classifier.py  --dataset tolokers
python ../src/train_classifier.py  --dataset questions
python ../src/train_classifier.py  --dataset YelpChi
python ../src/train_classifier.py  --dataset Cora
python ../src/train_classifier.py  --dataset BlogCatalog
python ../src/train_classifier.py  --dataset ACM


# Pretrain conditional diffusion models on all datasets
CLASS_COND=True
python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset reddit --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset reddit --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset reddit --class_cond $CLASS_COND
 
python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset weibo --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset weibo --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset weibo --class_cond $CLASS_COND
 
python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset tfinance --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset tfinance --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset tfinance --class_cond $CLASS_COND
 
python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset tolokers --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset tolokers --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset tolokers --class_cond $CLASS_COND
 
python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset questions --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset questions --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset questions --class_cond $CLASS_COND
 
python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset YelpChi --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset YelpChi --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset YelpChi --class_cond $CLASS_COND
 
python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset Cora --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset Cora --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset Cora --class_cond $CLASS_COND

python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset BlogCatalog --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset BlogCatalog --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset BlogCatalog --class_cond $CLASS

python ../src/train_graphdiffusion_cond.py --loss_type l1 --dataset ACM --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type l2 --dataset ACM --class_cond $CLASS_COND
python ../src/train_graphdiffusion_cond.py --loss_type huber --dataset ACM --class_cond $CLASS_COND



# Unconditional but guided sampling
CLASS_COND=False
CLASS_GUIDE=True

echo "starting class guided sampling"
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset reddit 
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset weibo
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset tolokers
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset questions
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset YelpChi

python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset Cora
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset BlogCatalog
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset ACM

# unconditional and unguided sampling
CLASS_COND=False
CLASS_GUIDE=False

echo "starting unguided sampling"
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset reddit 
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset weibo
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset tolokers
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset questions
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset YelpChi
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset Cora
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset BlogCatalog
python ../src/sampling.py --class_cond $CLASS_COND --class_guide $CLASS_GUIDE --dataset ACM