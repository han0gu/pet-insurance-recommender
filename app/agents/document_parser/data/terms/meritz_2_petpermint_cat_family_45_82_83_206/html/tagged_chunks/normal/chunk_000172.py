from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 해당 연<br>도의 계약해당일이 없는 경우에는 해당 월의 마지막 날을<br>계약해당일로 합니다.</p><br><p '
 "id='33' data-category='paragraph' style='font-size:16px'>예시1) 계약일 : 2020년 "
 '10월 1일<br>-> 계약해당일 : 10월 1일<br>예시2) 계약일 : 2020년 2월 29일<br>-> 계약해당일 : 2월 '
 "말일</p><h1 id='34' style='font-size:18px'>제25조(계약의 소멸)</h1><br><p id='35'"),
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
 'indexing': {'chunk_id': 'chunk_000172',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
