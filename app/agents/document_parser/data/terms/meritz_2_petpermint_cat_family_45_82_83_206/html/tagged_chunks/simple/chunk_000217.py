from langchain_core.documents import Document

chunk = Document(
    page_content=('따라 계약이 해지된 경우 회사는 제35<br>조(해약환급금) 제4항에 따른 해약환급금을 계약자에게 지<br>급합니다.<br>\uf000 '
 '계약자는 제1항의 제척기간에도 불구하고 민법 등 관계<br>법령에서 정하는 바에 따라 법률상의 권리를 행사 할 수 '
 "있<br>습니다.</p><br><h1 id='85' style='font-size:20px'>【위법계약】</h1><br><p "
 "id='86' data-category='paragraph' style='font-size:20px'>금융소비자보호에 관한 법률 "
 '제47조에서 정한 적합성원<br>칙,'),
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
 'indexing': {'chunk_id': 'chunk_000217',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
