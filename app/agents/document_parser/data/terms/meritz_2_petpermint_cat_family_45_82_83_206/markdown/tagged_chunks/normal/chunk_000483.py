from langchain_core.documents import Document

chunk = Document(
    page_content=('| 4 | 비뇨기과 | AGB001 | 신장의 악성 신생물 |\n'
 '| 4 | 비뇨기과 | AGC001 | 신장의 신생물 (양성 또는 악성이 불확실한) |\n'
 '| 4 | 비뇨기과 | AGB002 | 이행상피세포암종 (방광) |\n'
 '| 4 | 비뇨기과 | AGA003 | 기타 방광의 양성 신생물 기타 |\n'
 '| 4 | 비뇨기과 | AGB003 | 방광의 악성 신생물 |\n'
 '| 4 | 비뇨기과 |  |  |\n'
 '| 4 | 비뇨기과 | AGC003 | 기타 방광의 신생물 (양성 또는 악성이 불확 실한) |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000483',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
