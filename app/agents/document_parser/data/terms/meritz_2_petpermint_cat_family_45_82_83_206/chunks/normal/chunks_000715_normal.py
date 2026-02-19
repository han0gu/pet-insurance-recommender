from langchain_core.documents import Document

chunk = Document(
    page_content=('9) 손가락의 관절기능장해 평가는 손가락 관절의 관절운 동범위 제한 등으로 평가한다. 각 관절의 운동범위 측정은 장해평가시점의 '
 '｢산업재해보상보험법 시행규 칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 관절 에 대한 평균 운동가능영역을 기준으로 정상각도 및 '
 '측정방법 등을 따른다.\n'
 '< 손가락 >\n'
 '11. 발가락의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 한발의 리스프랑관절 이상을 잃었을 때 | 40\n'
 '2) 한발의 5개발가락을 모두 잃었을 때 | 30\n'
 '3) 한발의 첫째발가락을 잃었을 때 | 10'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 197},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000715',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
