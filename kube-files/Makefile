create-hf-token:
	kubectl delete -f 01-huggingface-secret.yaml
	kubectl create -f 01-huggingface-secret.yaml

model-download:
	kubectl delete -f 02-model-download-job.yaml
	kubectl apply -f 02-model-download-job.yaml

build-push-model-downloader-docker:
	docker build -t nehavikas/model-downloader:1 -f Dockerfile.downloader .
	docker push nehavikas/model-downloader:1
