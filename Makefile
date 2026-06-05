.PHONY: sync-internal clean-vectorstore

sync-internal:
ifndef msg
	$(error msg is required. Usage: make sync-internal msg="your message")
endif
	git submodule update --remote finthesis_internal
	git add finthesis_internal
	git commit -m "update submodule: $(msg)"

clean-vectorstore:
	@echo "Deleting all rows from documents table..."
	@python -c "\
import os; from dotenv import load_dotenv; load_dotenv(); \
from supabase import create_client; \
client = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_ROLE_KEY']); \
result = client.table('documents').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute(); \
print(f'Deleted {len(result.data)} chunks from documents table')"
