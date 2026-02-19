from langchain_core.documents import Document

chunk = Document(
    page_content=('② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제5조 (특별약관의 소멸)\n'
 '피보험자 또는 보험증권에 기재된 반려견이 보험기간 중에 사망하였을 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 '
 '회사가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없 습니다.\n'
 '제6조 (특별약관의 자동갱신)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 125},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000787',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
