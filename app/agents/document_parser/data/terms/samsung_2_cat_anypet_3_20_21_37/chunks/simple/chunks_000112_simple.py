from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이와 별도로 본 보험회사 보호상품의 사고보험금을 합산한 금액이 1인당 "1억원까지" 보호됩니다. 다만, 보험계약자 및 보 험료납부자가 '
 '법인인 보험계약의 경우에는 보호되지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 20},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000112',
              'chunk_char_len': 102,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
