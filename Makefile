.PHONY: sync-internal

sync-internal:
ifndef msg
	$(error msg is required. Usage: make sync-internal msg="your message")
endif
	git submodule update --remote finthesis_internal
	git add finthesis_internal
	git commit -m "update submodule: $(msg)"
