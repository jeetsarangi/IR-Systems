run:
	unzip 21111032-qrels.zip
	unzip 21111032-ir-systems.zip
	cd 21111032-ir-systems && sh mid.sh $(ARGS)
