from langchain_core.documents import Document

chunk = Document(
    page_content=('(피부)</td></tr><tr><td>AFA003</td><td>유두종 '
 '(피부)</td></tr><tr><td>AFA004</td><td>피지종</td></tr><tr><td>AFA005</td><td>모낭상피종</td></tr><tr><td>AFA006 '
 'AFA007</td><td>기저세포종 비만세포종 (피부) (양성)</td></tr><tr><td>AFB007</td><td>비만세포종 '
 '(피부) (악성)</td></tr><tr><td>AFC007</td><td>비만세포종(피부) (양성 또는 악성이 불확실'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000871',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
