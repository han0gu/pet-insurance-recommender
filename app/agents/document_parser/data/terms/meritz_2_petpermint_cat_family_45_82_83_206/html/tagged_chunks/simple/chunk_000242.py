from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험금 지급사유가 2023년 4월 1일에 발생하였음에<br>도 2026년 4월 1일까지 보험금을 청구하지 않는 경우 소<br>멸시효가 '
 "완성되어 보험금 등을 지급받지 못할 수 있습니<br>다.</p><h1 id='32' "
 "style='font-size:18px'>제42조(약관의 해석)</h1><br><p id='33' "
 "data-category='paragraph' style='font-size:18px'>\uf000 회사는 신의성실의 원칙에 따라 "
 '공정하게 약관을 해석하<br>여야 하며 계약자에 따라 다르게 해석하지'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000242',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
