from langchain_core.documents import Document

chunk = Document(
    page_content=('일 현재 유지중이거나, 계약 청약일 전 6개월 이내에 계약자 및 피보험자의 요구 또는\n'
 '보험료 납입 연체로 해지된 경우 유사계약에서 정한 부담보 기간 종료일 이내에서 계\n'
 '약의 부담보 기간을 적용하고, 유사계약에서 정한 질병과 동일하거나 축소된 범위로\n'
 '계약의 부담보 설정 범위를 정하며, 유사계약이 다수인 경우 피보험자에게 가장 유리\n'
 '한 계약조건을 적용합니다. 단, 유사계약 청약일 이후 제1항 제1호 또는 제2호에서 정\n'
 '한 질병과 관련한 새로운 위험(재진단·치료 등은 해당하지 않습니다)이 발생하거나,'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000561',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
