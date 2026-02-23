from langchain_core.documents import Document

chunk = Document(
    page_content=('는 것을 말합니다. 단 수술에서 아래에 정한 사항은 제외합니다- 1. 흡인 (주사기 등으로 빨아 들이는 것)\n'
 '- 2. 천자 (바늘 또는 관을 꽂아 체액, 조직을 뽑아내거나 약물을 주입하는 것) 등의 조치\n'
 '- 3. 미용성형 목적의 수술\n'
 '- 4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)\n'
 '④ 회사가 지급할 제1항에서 정한 보험금은 피보험자가 부담한 수술 당일 발생한 의료비\n'
 '에서 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000547',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
