RUNTEST=python -m unittest discover -v -b

all:
	${RUNTEST} parsing
	${RUNTEST} backend
