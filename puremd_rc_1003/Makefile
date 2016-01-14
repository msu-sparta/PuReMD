all: cpu gpu parallel-cpu parallel-gpu

gpu:
	@echo "** Invoking ./PuReMD-GPU/Makefile"
	($(MAKE) -C ./PuReMD-GPU)

parallel-gpu:
	@echo "** Invoking ./PG-PuReMD/Makefile"
	($(MAKE) -C ./PG-PuReMD)

cpu:
	@echo "** Invoking ./sPuReMD/Makefile"
	($(MAKE) -C ./sPuReMD)

parallel-cpu:
	@echo "** Invoking ./PuReMD/Makefile"
	($(MAKE) -C ./PuReMD)

clean:
	@echo "** cleaning ./PuReMD-GPU"
	($(MAKE) -C ./PuReMD-GPU clean)
	@echo "** cleaning ./PG-PuReMD"
	($(MAKE) -C ./PG-PuReMD clean)
	@echo "** cleaning ./sPuReMD"
	($(MAKE) -C ./sPuReMD clean)
	@echo "** cleaning ./PuReMD"
	($(MAKE) -C ./PuReMD clean)
