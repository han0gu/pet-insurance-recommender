from langchain_core.documents import Document

chunk = Document(
    page_content=("적용에 관한 세부지침」을 따릅니다.</p><br><p id='72' data-category='paragraph' "
 "style='font-size:20px'>【적립부분 적립이율】</p><br><p id='73' "
 "data-category='paragraph' style='font-size:20px'>적립부분 계약자적립액 계산시 적립부분 순보험료에 "
 "대<br>한 이자를 계산할 때 적용하는 이율을 말합니다.</p><footer id='74' "
 "style='font-size:14px'>54</footer><h1 id='75'"),
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
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
