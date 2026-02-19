from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 계약자가 사망한 이후 그 승계인이 보험 수익자를 변경할 수 있다는 별도의 약정이 있는 경우에는 승계받은 계약자가 보험수 익자를 '
 '변경할 수 있습니다. ⑥ 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관을 드리고, 변경된 계약자가 '
 '요청하는 경우 약관의 중요한 내용을 설명하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000269',
              'chunk_char_len': 181,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
