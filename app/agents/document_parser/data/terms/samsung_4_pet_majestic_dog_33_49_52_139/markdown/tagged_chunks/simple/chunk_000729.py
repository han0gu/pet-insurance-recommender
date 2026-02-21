from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 질병과 관련한 새로운 위험(재진단·치료 등은 해당하지 않습니다)이 발생하거나,\n'
 '- 새로운 질병에 대한 보장이 추가(입원비, 수술비, 진단비 등 보장 범위의 변경 또는 확\n'
 '- 대는 해당하지 않습니다)된 경우 이를 적용하지 아니할 수 있습니다.\n'
 '- ④ 제2항에서 부담보 기간을「보험계약의 보험기간 전체」로 적용한 경우 최초 계약 청\n'
 '- 약일부터 5년 이내에 제1항 제1호 또는 제2호에서 정한 질병으로 재진단 또는 치료를\n'
 '- 받지 않은 경우에는 최초 계약 청약일부터 5년이 지난 이후에는 이 특별약관을 적용'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000729',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
