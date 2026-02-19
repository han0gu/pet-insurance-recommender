from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 한 눈의 안구(눈동자)의 주시야(머리를 움직이 지 않고 눈만을 움직여서 볼 수 있는 범위)의 운동범위가 정상의 1/2 이하로 감소된 '
 '경우 나) 중심 20도 이내에서 복시(물체가 둘로 보이거나 겹쳐 보임)를 남긴 경우'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 178},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000631',
              'chunk_char_len': 123,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
