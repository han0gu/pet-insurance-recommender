from langchain_core.documents import Document

chunk = Document(
    page_content=('제 5조 (특별약관의 소멸)\n'
 '피보험자가 보험기간 중에 사망하였을 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사가 적립한 사망당시 이 '
 '특별약관의 계약자적립액 및 미경과보험료 를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000487',
              'chunk_char_len': 146,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
