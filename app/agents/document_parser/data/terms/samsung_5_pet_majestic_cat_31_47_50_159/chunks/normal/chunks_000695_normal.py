from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사가 제1조(보험금의 지급사유) 제1항에서 정한 사망보험금을 지급한 때에는 그 손 해보상의 원인이 생긴 때부터 이 특별약관은 '
 '소멸되며 그 때부터 효력이 없습니다. 이 경우 회사는 이 특별약관의 해약환급금을 지급하지 않습니다. ② 보험증권에 기재된 반려묘가 '
 '보험기간 중에 이 특별약관에서 보장하지 않는 사유로 사망하였을 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사 가 '
 '적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료를 계약자에게 지급 하고, 이 특별약관은 더 이상 효력이 없습니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 112},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000695',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
