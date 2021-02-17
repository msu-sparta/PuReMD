FC=gfortran
FFLAGS=-std=f2003
LDFLAGS=-L../../../../sPuReMD/lib/.libs -lspuremd
PROGRAM=driver
DEPS=$(patsubst %.F03, %.o, $(wildcard *.F03))

ifeq ($(V),1)
ECHO := 
else
ECHO := @
endif

all: $(PROGRAM)

$(PROGRAM): $(DEPS)
	$(ECHO)$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.F03
	$(ECHO)$(FC) $(FFLAGS) -c -o $@ $<

clean:
	$(ECHO)rm -f $(PROGRAM) $(DEPS) *.mod

debug:
	@echo "PROGRAM = $(PROGRAM)"
	@echo "DEPS = $(DEPS)"
	@echo "RC = $(FC)"
	@echo "FFLAGS = $(FFLAGS)"
	@echo "LDFLAGS = $(LDFLAGS)"

.PHONY: all clean
