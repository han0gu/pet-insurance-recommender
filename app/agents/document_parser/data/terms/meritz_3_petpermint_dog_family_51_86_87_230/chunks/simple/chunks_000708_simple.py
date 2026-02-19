from langchain_core.documents import Document

chunk = Document(
    page_content=('7) “안구(눈동자)의 뚜렷한 조절기능장해“라 함은 조 절력이 정상의 1/2 이하로 감소된 경우를 말한다. 다 만, 조절력의 감소를 '
 '무시할 수 있는 50세 이상(장해 진단시 연령 기준)의 경우에는 제외한다. 8) “뚜렷한 시야 장해”라 함은 한 눈의 시야 범위가 '
 '정상시야 범위의 60% 이하로 제한된 경우를 말한다. 이 경우 시야검사는 공인된 시야검사방법으로 측정 하며, 시야장해 평가 시 '
 '자동시야검사계(골드만 시 야검사)를 이용하여 8방향 시야범위 합계를 정상범 위와 비교하여 평가한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 203},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000708',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
