from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기간을 의미합니다.\n'
 '- ⑦ 제1항의 「검사비」란 아래에 정한 검사에 사용된 비용을 의미합니다. 단, 제6조(보험\n'
 '- 금을 지급하지 않는 사유) 제2항의 의료비 및 비용을 위한 검사는 제외합니다.\n'
 '- 1. 신체검사, 정형검사, 신경계검사, 안검사, 피부과검사 등 기본검사\n'
 '- 2. X-ray, 초음파검사, CT, MRI, 내시경검사 등 영상검사\n'
 '- 3. 혈액검사, 임상병리검사, 조직병리검사, 배양검사 등 실험실검사\n'
 '- ⑧ 제2항에도 불구하고 제27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라'),
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
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000301',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
