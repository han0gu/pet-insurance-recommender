from langchain_core.documents import Document

chunk = Document(
    page_content=('상완골에 가관절이 남은 경우 또는 요골과 척골의 2\n'
 '개뼈 모두에 가관절이 남은 경우를 말한다.※ 주) 가관절이란, 충분한 경과 및 골이식술 등 골\n'
 '유합을 얻는데 필요한 수술적 치료를 시행하\n'
 '였음에도 불구하고 골절부의 유합이 이루어\n'
 '지지 않는 ‘불유합’ 상태를 말하며, 골유\n'
 '합이 지연되는 지연유합은 제외한다.- 12) “가관절이 남아 약간의 장해를 남긴 때”라 함은\n'
 '- 요골과 척골중 어느 한 뼈에 가관절이 남은 경우를\n'
 '- 말한다.\n'
 '- 13) “뼈에 기형을 남긴 때”라 함은 상완골 또는 요골'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000575',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
