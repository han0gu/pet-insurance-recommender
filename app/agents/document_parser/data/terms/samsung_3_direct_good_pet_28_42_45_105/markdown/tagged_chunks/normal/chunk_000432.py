from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 천자 (바늘 또는 관을 꽂아 체액, 조직을 뽑아내거나 약물을 주입하는 것) 등의 조치\n'
 '- 3. 미용성형 목적의 수술\n'
 '- 4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)\n'
 '- ④ 제1항 내지 제3항에도 불구하고 이 특별약관의 보험계약일부터 그 날을 포함하여 1년\n'
 '- 이내에 발생한 슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성부전 또는 기타\n'
 '- 이들과 유사한 사고에 대해서는 보험금을 지급하지 않습니다. 단, 이 계약이 제27조 ('),
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
 'indexing': {'chunk_id': 'chunk_000432',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
