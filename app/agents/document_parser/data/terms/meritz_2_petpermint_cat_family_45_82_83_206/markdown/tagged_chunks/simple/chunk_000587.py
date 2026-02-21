from langchain_core.documents import Document

chunk = Document(
    page_content=('- 골과 종아리뼈중 어느 한 뼈에 가관절이 남은 경우\n'
 '- 를 말한다.\n'
 '- 14) "뼈에 기형을 남긴 때"라 함은 대퇴골 또는 경골에\n'
 '- 기형이 남아 정상에 비해 부정유합된 각 변형이\n'
 '- 15° 이상인 경우를 말한다.\n'
 '- 15) 다리 길이의 단축 또는 과신장은 스캐노그램\n'
 '- (scanogram)을 통하여 측정한다.\n'
 '# 다. 지급률의 결정- 1) 한 다리의 3대 관절중 관절 하나에 기능장해가 생기고\n'
 '- 다른 관절 하나에 기능장해가 발생한 경우 지급률은\n'
 '- 각각 적용하여 합산한다.'),
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
            'risk_domains': ['digestive', 'joint', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000587',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
