from langchain_core.documents import Document

chunk = Document(
    page_content=('| 외부요인에 의한 폐질환 | 탄광부 진폐증 | J60 공 |\n'
 '| 외부요인에 의한 폐질환 | 석면 및 기타 광섬유에 의한 진폐증 | J61 통 |\n'
 '| 외부요인에 의한 폐질환 | 실리카를 함유한 먼지에 의한 진폐증 | J62 사항 |\n'
 '| 외부요인에 의한 폐질환 | 기타 무기물먼지에 의한 진폐증 | J63 |\n'
 '| 외부요인에 의한 폐질환 | 상세불명의 진폐증 | J64 |\n'
 '| 외부요인에 의한 폐질환 | 결핵과 연관된 진폐증 | J65 |\n'
 '| 외부요인에 의한 폐질환 | 특정 유기물먼지에 의한 기도질환 | J66 보 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001020',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
