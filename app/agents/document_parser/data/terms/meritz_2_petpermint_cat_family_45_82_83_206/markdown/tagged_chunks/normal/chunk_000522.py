from langchain_core.documents import Document

chunk = Document(
    page_content=('- 8) “뚜렷한 시야 장해”라 함은 한 눈의 시야 범위가\n'
 '- 정상시야 범위의 60% 이하로 제한된 경우를 말한다.\n'
 '- 이 경우 시야검사는 공인된 시야검사방법으로 측정\n'
 '- 하며, 시야장해 평가 시 자동시야검사계(골드만 시\n'
 '- 야검사)를 이용하여 8방향 시야범위 합계를 정상범\n'
 '- 위와 비교하여 평가한다.\n'
 '- 9) “눈꺼풀에 뚜렷한 결손을 남긴 때”라 함은 눈꺼풀\n'
 '- 의 결손으로 눈을 감았을 때 각막(검은 자위)이 완전\n'
 '- 히 덮이지 않는 경우를 말한다.\n'
 '178- 10) “눈꺼풀에 뚜렷한 운동장해를 남긴 때“라 함은 눈'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000522',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
