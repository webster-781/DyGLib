new_datasets=('SMS-A' 'comm-linux-kernel-reply' 'ia-chess' 'digg-friends' 'ia-enron-email-all' 'imdb' 'rec-stackoverflow' 'soc-sign-bitcoinalpha' 'soc-sign-bitcoinotc' 'ia-stackexch-user-marks-post-und' 'soc-youtube-growth' 'tech-as-topology')
cd ~/DyGLib/preprocess_data
for dataset in "${new_datasets[@]}"
do
    # wget https://nrvis.com/download/data/dynamic/${dataset}.zip
    # unzip ${dataset}.zip -d ${dataset}
    echo ${dataset}
    echo --------------------------------------------------------------------
    python preprocess_data.py --dataset_name ${dataset}
    # head ${dataset}/${dataset}.csv -n 10
    echo --------------------------------------------------------------------
done