from langchain_core.documents import Document

chunk = Document(
    page_content=('신생물</td></tr><tr><td>AIB001</td><td>눈 및 부속 기관의 악성 '
 '신생물</td></tr><tr><td>AIC001</td><td>눈 및 부속 기관의 신생물 (양성 또는 악성이 '
 '불확실한)</td></tr><tr><td>FAA001</td><td>안검 '
 '외반</td></tr><tr><td>FAA002</td><td>안검 '
 '내반</td></tr><tr><td>FAA003</td><td>안검염</td></tr><tr><td>FAA004</td><td>다래끼 / '
 '산립종 /'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000853',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
