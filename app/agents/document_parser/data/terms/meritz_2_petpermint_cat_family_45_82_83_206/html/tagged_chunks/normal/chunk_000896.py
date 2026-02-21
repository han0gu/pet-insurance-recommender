from langchain_core.documents import Document

chunk = Document(
    page_content=(". 별표</header><h1 id='23' style='font-size:16px'>【별표1】</h1><p id='24' "
 "data-category='paragraph' style='font-size:18px'>보험금을 지급할 때의 적립이율 계산<br>(제8조 "
 "제5항, 제10조 제3항 및 제35조 제2항 관련)</p><table id='25' "
 "style='font-size:16px'><thead><tr><td>구 분</td><td>기 간</td><td>지 급 이 "
 '자</td></tr></thead><tbody><tr><td'),
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
 'indexing': {'chunk_id': 'chunk_000896',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
