from langchain_core.documents import Document

chunk = Document(
    page_content=('- 받지 않은 경우에는 최초 계약 청약일부터 5년이 지난 이후에는 이 특별약관을 적용\n'
 '- 하지 않습니다. 단, 계약 청약일 현재 부담보 기간을 「보험계약의 보험기간 전체」로\n'
 '- 적용한 유사계약이 유지중이거나, 계약 청약일 전 6개월 이내에 계약자 및 피보험자\n'
 '- 의 요구 또는 보험료 납입 연체로 해지된 경우 유사계약 청약일과 계약 청약일 사이\n'
 '- 에 제1항 제1호 또는 제2호에서 정한 질병으로 재진단 또는 치료를 받지 않았다면 계'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000695',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
