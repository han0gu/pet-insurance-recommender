from langchain_core.documents import Document

chunk = Document(
    page_content=("장소)</h1><br><p id='47' data-category='paragraph' "
 "style='font-size:20px'>\uf000 이 계약에 있어서「수술」이라 함은 수의사가 치료가 필<br>요하다고 인정한 경우로서 "
 '수의사의 관리하에 치료를 직접<br>적인 목적으로 기구를 사용하여 생체(生體)에 절단, 절제<br>등의 조작을 가하는 것을 말합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000592',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
