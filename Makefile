LRC ?= results/test_song.lrc
STYLE ?=

.PHONY: help video

help:
	@echo "make video LRC=path/to/song.lrc STYLE='Rock, Piano'"

video:
	./generator.sh "$(LRC)" "$(STYLE)"
