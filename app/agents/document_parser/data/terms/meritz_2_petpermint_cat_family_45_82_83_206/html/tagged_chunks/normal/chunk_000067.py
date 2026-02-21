from langchain_core.documents import Document

chunk = Document(
    page_content=('4월 1일</td><td>2천만원</td></tr><tr><td>2026년 4월 1일</td><td>2천만원 × (1 + '
 '평균공시이율)</td></tr><tr><td>2027년 4월 1일</td><td>2천만원 × (1 + '
 "평균공시이율)2</td></tr></tbody></table><p id='92' data-category='paragraph' "
 "style='font-size:20px'>2"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000067',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
