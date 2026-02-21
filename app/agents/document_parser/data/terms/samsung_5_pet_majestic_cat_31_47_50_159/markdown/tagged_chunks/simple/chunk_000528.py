from langchain_core.documents import Document

chunk = Document(
    page_content=('지를 받은 날을 말합니다)부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.# 제24조 (계약자의 임의해지)계약자는 특별약관이 '
 '소멸하기 전에는 언제든지 이 특별약관을 해지할 수 있으며, 이 경\n'
 '우 회사는 이 특별약관의 해약환급금을 계약자에게 지급합니다. 다만, 타인을 위한 계약\n'
 '의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여 특별\n'
 '약관을 해지할 수 있습니다.# 제25조 (중대사유로 인한 해지)① 회사는 아래와 같은 사실이 있을 경우에는 그 사실을 안 날부터 1개월 '
 '이내에 이 특별'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000528',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
