.PHONY: sync-internal

sync-internal:
ifndef msg
	$(error msg is required. Usage: make sync-internal msg="your message")
endif
	cd finthesis_internal && git add . && git commit -m "$(msg)" && git push
	git add finthesis_internal
	git commit -m "update submodule: $(msg)"
