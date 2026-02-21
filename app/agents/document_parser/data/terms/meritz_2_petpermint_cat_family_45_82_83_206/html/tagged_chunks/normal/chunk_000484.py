from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단,「반려동물 비용손해 관련 특별약관 일반<br>조항」제15조(재가입) 제6항에 따라 보험계약이 연장된 경<br>우에는 적용하지 '
 '않습니다.<br>\uf000 제1항의「연간」이라 함은 계약일부터 매 1년 단위로 도<br>래하는 계약해당일 전일까지의 기간을 '
 '말합니다.<br>\uf000 반려동물이 제1항의 질병 또는 상해로 치료를 받던 중에<br>보험기간이 만료된 경우에도 만료일부터 180일 '
 '이내의 치료<br>비는 제2항에 따라 보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000484',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
