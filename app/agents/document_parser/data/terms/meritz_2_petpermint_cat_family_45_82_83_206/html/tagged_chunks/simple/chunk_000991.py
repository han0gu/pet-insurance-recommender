from langchain_core.documents import Document

chunk = Document(
    page_content=('척추체(척추뼈 몸통)에 골절 또는 탈구로 3개의<br>척추체(척추뼈 몸통)를 유합(아물어 붙음) 또는<br>고정한 상태<br>나) '
 '머리뼈(두개골)와 제1경추 또는 제1경추와 제2경<br>추를 유합 또는 고정한 상태<br>다) 머리뼈(두개골)와 상위목뼈(상위경추: '
 '제1, 2경<br>추) 사이에 CT 검사 상, 두개 대후두공의 기저점<br>(basion)과 축추 치돌기 상단사이의 거리(BDI '
 ':<br>Basion-Dental Interval)에 뚜렷한 이상전위가<br>있는 상태<br>라) 상위목뼈(상위경추: 제1, 2경추) '
 'CT'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000991',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
