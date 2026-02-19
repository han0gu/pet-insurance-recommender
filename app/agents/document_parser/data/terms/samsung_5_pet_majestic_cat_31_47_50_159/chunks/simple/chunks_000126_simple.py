from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[보험계약대출이율]\n'
 '해당 보험상품의 약관에 따라 계약자가 대출을 받을 경우, 회사가 정하는 대출이율이며, 이 계약의 보험계약대출이율이 변경되는 경우, 변경된 '
 '시점부터 변경된 이율을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 42},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000126',
              'chunk_char_len': 114,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
