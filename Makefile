# コンテナ立ち上げ
up:
	docker-compose up py

# とりあえずpython実行できるか
test:
	docker-compose exec py /bin/bash -c "python test.py"

# pip install
pip:
	docker-compose exec py /bin/bash -c "pip install -r requrements.txt"


# 学習を開始する
retrain:
	python retrain.py --image_dir images/flower_photos
# 学習を開始する(mobilenet)
retrain_mobile:
	python retrain.py --image_dir images/flower_photos --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1

# 実行
labels:
	python label_image.py --graph=./tmp/output_graph.pb --labels=./tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=images/flower_photos/daisy/21652746_cc379e0eea_m.jpg
labels_mobile:
	python label_image.py --graph=./tmp/output_graph.pb --labels=./tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --input_height=224 --input_width=224 --image=images/flower_photos/daisy/21652746_cc379e0eea_m.jpg
