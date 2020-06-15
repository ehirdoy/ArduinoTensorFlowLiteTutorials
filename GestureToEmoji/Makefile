run:
	docker run -it --rm -u $(shell id -u):$(shell id -g) -v $(shell pwd):/work -w /work mytf python train.py
	echo "const unsigned char model[] = {" > model.h
	cat gesture_model.tflite | xxd -i      >> model.h
	echo "};"                              >> model.h
	mv model.h ArduinoSketches/IMU_Classifier/model.h
	git diff
build:
	docker build . -t mytf

