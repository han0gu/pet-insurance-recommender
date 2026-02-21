from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금을 지급하지 않는 사유에 해당하는 경우 등을 말합니다.⑧ 회사가 지급하여야 할 하나의 상해로 인한 상해 후유장해보험금은 상해 '
 '후유장해보험\n'
 '가입금액을 한도로 합니다.- 67 -# 제 3조 (특별약관의 소멸)피보험자가 보험기간 중에 사망하였을 경우에는 "보험료 및 해약환급금 '
 '산출방법서"에서\n'
 '정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000299',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
