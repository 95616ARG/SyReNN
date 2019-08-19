.PHONY: docker-image docker-box

IMAGE_NAME=masotoud-vrl-train

docker-image:
	docker build --force-rm -t $(IMAGE_NAME) .

test_pendulum: docker-image
	docker run --rm -t -i --user="$(id -u):$(id -g)" -v $(PWD):/iovol \
	    -w /vrl $(IMAGE_NAME) \
	    make test_pendulum

savenet:
	docker run --rm -t -i --user="$(id -u):$(id -g)" -v $(PWD):/iovol \
	    -w /vrl $(IMAGE_NAME) \
	    make savenet MODEL=$(MODEL)

retrain_and_test_pendulum: docker-image
	docker run --rm -t -i --user="$(id -u):$(id -g)" -v $(PWD):/iovol \
	    -w /vrl $(IMAGE_NAME) \
	    make retrain_and_test_pendulum
