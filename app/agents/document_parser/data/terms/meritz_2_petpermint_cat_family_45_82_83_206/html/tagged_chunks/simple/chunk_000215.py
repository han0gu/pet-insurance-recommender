from langchain_core.documents import Document

chunk = Document(
    page_content=("제1항에 따른 해약환급금을 계약자에게 지급합니다.</p><h1 id='83' "
 "style='font-size:16px'>제32조의1(위법계약의 해지)</h1><br><p id='84' "
 "data-category='paragraph' style='font-size:20px'>\uf000 계약자는 ｢금융소비자보호에 관한 "
 '법률｣ 제47조 및 관련<br>규정이 정하는 바에 따라 계약체결에 대한 회사의 법위반사<br>항이 있는 경우 계약체결일부터 5년 이내의 '
 '범위에서 계약<br>자가 위반사항을 안 날로부터 1년 이내에 계약해지요구서에<br>증빙서류를'),
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
 'indexing': {'chunk_id': 'chunk_000215',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
