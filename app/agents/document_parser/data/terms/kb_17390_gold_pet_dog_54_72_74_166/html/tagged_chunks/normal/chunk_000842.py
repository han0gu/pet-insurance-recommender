from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약<br>자 또는 피보험자가 사실대로 알리는 것을 방해한 경우, '
 '계약자 또는 피보험<br>자에게 사실대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을<br>때'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000842',
              'chunk_char_len': 135,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
