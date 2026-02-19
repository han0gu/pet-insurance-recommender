from langchain_core.documents import Document

chunk = Document(
    page_content=('13) "가관절이 남아 약간의 장해를 남긴 때"라 함은 경 골과 종아리뼈중 어느 한 뼈에 가관절이 남은 경우 를 말한다. 14) "뼈에 '
 '기형을 남긴 때"라 함은 대퇴골 또는 경골에 기형이 남아 정상에 비해 부정유합된 각 변형이 15° 이상인 경우를 말한다. 15) 다리 '
 '길이의 단축 또는 과신장은 스캐노그램 (scanogram)을 통하여 측정한다.\n'
 '다. 지급률의 결정'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 195},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000707',
              'chunk_char_len': 205,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
