from langchain_core.documents import Document

chunk = Document(
    page_content=('⑧ 제2항에도 불구하고 제27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라 재가입하는 경우 또는 제27조 (특별약관의 '
 '재가입에 관한 사항) 제5항에 따라 보험계 약이 연장된 경우에는 종전 계약의 보험기간을 연장하는 것으로 보아 제2항을 적용하 지 '
 '않습니다. ⑨ 제3항에도 불구하고 제27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라 재가입하는 경우 또는 제27조 '
 '(특별약관의 재가입에 관한 사항) 제5항에 따라 보험계 약이 연장된 경우에는 보장개시일(책임개시일)은 이 특별약관의 보험계약일로 봅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000350',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
