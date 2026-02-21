from langchain_core.documents import Document

chunk = Document(
    page_content=('뚜렷한 이상전위가<br>있는 상태<br>라) 상위목뼈(상위경추: 제1, 2경추) CT 검사상, 환<br>추 전방 궁(arch)의 후방과 '
 '치상돌기의 전면과의<br>거리(ADI: Atlanto-Dental Interval)에 뚜렷한<br>이상전위가 있는 상태</p><br><p '
 "id='3' data-category='list' style='font-size:20px'>8) 약간의 운동장해<br>머리뼈(두개골)와 "
 '상위목뼈(상위경추: 제1, 2경추)를<br>제외한 척추체(척추뼈 몸통)에 골절 또는 탈구로 2개<br>의 척추체(척추뼈'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000992',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
