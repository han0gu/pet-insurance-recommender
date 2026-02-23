from langchain_core.documents import Document

chunk = Document(
    page_content=('- 슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성부전 또는 기타 이들과 유사한\n'
 '- 사고에 대해서는 보험금을 지급하지 않습니다. 단, 이 계약이 제27조 (특별약관의 재\n'
 '- 가입에 관한 사항) 제1항 및 제2항에 따라 재가입하는 경우 또는 제27조 (특별약관의\n'
 '- 재가입에 관한 사항) 제5항에 따라 보험계약이 연장된 경우에는 적용하지 않습니다.\n'
 '- ⑤ 회사가 지급할 제1항에서 정한 의료비보험금(수술당일제외, 검사비포함)은 보험증권에\n'
 '- 기재된 자기부담금을 차감한 후 보상비율을 곱한 금액이며 보험증권에 기재된 1일당'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000298',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
