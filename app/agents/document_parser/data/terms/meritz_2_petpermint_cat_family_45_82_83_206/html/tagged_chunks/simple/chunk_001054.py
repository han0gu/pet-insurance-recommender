from langchain_core.documents import Document

chunk = Document(
    page_content=('다른 네 손가락의 제1지관절(근위지관<br>절)부터 심장에서 먼쪽으로 손가락 뼈의 일부가 절<br>단된 경우를 말하며, 뼈 단면이 '
 '불규칙해진 상태나<br>손가락 길이의 단축 없이 골편만 떨어진 상태는 해당<br>하지 않는다.<br>7) “손가락에 뚜렷한 장해를 남긴 '
 '때”라 함은 첫째 손가<br>락의 경우 중수지관절 또는 지관절의 굴신(굽히고 펴<br>기)운동영역이 정상 운동영역의 1/2 이하인 경우를 '
 '말<br>하며, 다른 네 손가락에서는 제1, 제2지관절의 굴신<br>(굽히고 펴기)운동영역을 합산하여 정상운동영역의<br>1/2'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001054',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
