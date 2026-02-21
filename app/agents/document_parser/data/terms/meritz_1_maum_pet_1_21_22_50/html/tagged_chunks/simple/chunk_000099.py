from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험설계사 등이 계약자 또는 피보험자에게 고지할 기회를 주지 않았거나 계약자 또<br>는 피보험자가 사실대로 고지하는 것을 방해한 '
 '경우, 계약자 또는 피보험자에게 사<br>실대로 고지하지 않게 하였거나 부실한 고지를 권유했을 때'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
