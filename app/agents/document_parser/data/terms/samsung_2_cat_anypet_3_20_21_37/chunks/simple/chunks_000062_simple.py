from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 제12조(계약 전 알릴 의무)의 규정에 의하여 계약자 또는 피보험자가 회사에 알린 내용이 보험 금 지급사유의 발생에 영향을 미쳤음을 '
 '회사가 증명하는 경우 2. 제5조(보상하지 않는 손해), 제14조(사기에 의한 계약), 제18조(계약의 무효) 또는 제26조(계약 의 '
 '해지)의 규정을 준용하여 회사가 보장을 하지 않을 수 있는 경우'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
