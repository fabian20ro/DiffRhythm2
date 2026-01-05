LRC ?= results/test_song.lrc
STYLE ?=

.PHONY: help video

help:
	@echo "make video LRC=path/to/song.lrc STYLE='Rock, Piano'"

setup:
	python3 -m venv .venv
	. .venv/bin/activate
	pip install -r requirements.txt

video:
	./generator.sh "$(LRC)" "$(STYLE)"
