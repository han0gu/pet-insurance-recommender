from langchain_core.documents import Document

chunk = Document(
    page_content=('이들과 유사한 질병 또는 상해에 대해서는 보험금을 지급하\n'
 '지 않습니다. 단,「반려동물 비용손해 관련 특별약관 일반\n'
 '조항」제15조(재가입) 제6항에 따라 보험계약이 연장된 경\n'
 '우에는 적용하지 않습니다.\n'
 '\uf000 제1항의「연간」이라 함은 계약일부터 매 1년 단위로 도\n'
 '래하는 계약해당일 전일까지의 기간을 말합니다.\n'
 '\uf000 반려동물이 제1항의 질병 또는 상해로 치료를 받던 중에\n'
 '보험기간이 만료된 경우에도 만료일부터 180일 이내의 치료\n'
 '비는 제2항에 따라 보상하여 드립니다. 다만, 사고일 또는\n'
 '발병일부터 365일 이내인 경우에 한합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000393',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
