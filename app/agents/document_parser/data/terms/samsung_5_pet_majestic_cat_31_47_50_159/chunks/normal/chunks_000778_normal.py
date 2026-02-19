from langchain_core.documents import Document

chunk = Document(
    page_content=('제8조 (특별약관의 소멸)\n'
 '보험증권에 기재된 반려묘가 보험기간 중에 사망하여 이 추가특별약관에서 정한 보험금 지급사유가 더이상 발생할 수 없는 경우에는 "보험료 및 '
 '해약환급금 산출방법서" 에 정 하는 바에 따라 회사가 적립한 사망당시 이 추가특별약관의 계약자적립액 및 미경과보험 료를 계약자에게 '
 '지급하고, 이 추가특별약관은 더 이상 효력이 없습니다.\n'
 '제9조 (준용규정)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 120},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000778',
              'chunk_char_len': 206,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
