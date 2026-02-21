from langchain_core.documents import Document

chunk = Document(
    page_content=('장 질환</td></tr><tr><td>KDA010</td><td>염증성 장 '
 '질환(IBD)</td></tr><tr><td>KDA011</td><td>단백 소실성 '
 '장증(PLE)</td></tr><tr><td>KDA012</td><td>장폐색</td></tr><tr><td>KDA013</td><td>변비 '
 '(거대결장증 포함)</td></tr><tr><td>KDA014</td><td>모구증 (헤어볼 '
 '질환)</td></tr><tr><td>KDA015</td><td>장중첩</td></tr></tbody></table><footer'),
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
 'indexing': {'chunk_id': 'chunk_000885',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
