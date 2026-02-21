from langchain_core.documents import Document

chunk = Document(
    page_content=('굴신(굽히고 펴<br>기)운동범위 합계가 정상 운동가능영역의 1/2 이하가<br>된 경우를 말하며, 다른 네 발가락에 있어서는 '
 '중족지<br>관절의 신전운동범위만을 평가하여 정상운동범위의 1/2<br>이하로 제한된 경우를 말한다.<br>7) 한 발가락에 장해가 '
 '생기고 다른 발가락에 장해가 발<br>생한 경우, 지급률은 각각 적용하여 합산한다.<br>8) 발가락 관절의 운동범위 측정은 '
 '장해평가시점의 ｢산업<br>재해보상보험법 시행규칙｣ 제47조 제1항 및 제3항의 정<br>상인의 신체 각 관절에 대한 평균 운동가능영역을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001067',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
