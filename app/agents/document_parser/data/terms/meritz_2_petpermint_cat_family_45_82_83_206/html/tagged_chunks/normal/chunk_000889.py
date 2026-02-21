from langchain_core.documents import Document

chunk = Document(
    page_content=('증</td></tr><tr><td>KGA003</td><td>콕시듐증</td></tr><tr><td>KGA004</td><td>회충증</td></tr><tr><td>KGA005</td><td>촌충증</td></tr><tr><td>KGA006</td><td>간충증</td></tr><tr><td>KGA007</td><td>기타 '
 '소화기계 기생충증</td></tr><tr><td>KGA008</td><td>기타 소화기계 '
 '감염증</td></tr><tr><td>KGA009</td><td>소화계통의 기타'),
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
 'indexing': {'chunk_id': 'chunk_000889',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
