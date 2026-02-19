from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제2항에서 부담보 기간을「보험계약의 보험기간 전체」로 적용한 경우 최초 계약 청 약일부터 5년 이내에 제1항 제1호 또는 '
 '제2호에서 정한 질병으로 재진단 또는 치료를 받지 않은 경우에는 최초 계약 청약일부터 5년이 지난 이후에는 이 특별약관을 적용 하지 '
 '않습니다. 단, 계약 청약일 현재 부담보 기간을 「보험계약의 보험기간 전체」로 적용한 유사계약이 유지중이거나, 계약 청약일 전 6개월 '
 '이내에 계약자 및 피보험자'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000653',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
