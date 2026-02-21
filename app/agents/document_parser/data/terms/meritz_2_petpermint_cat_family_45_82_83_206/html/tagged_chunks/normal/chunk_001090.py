from langchain_core.documents import Document

chunk = Document(
    page_content=('평가한다.<br>그러나, 12개월이 지났다고 하더라도 뚜렷하게<br>기능 향상이 진행되고 있는 경우 또는 단기간내<br>에 사망이 '
 '예상되는 경우는 6개월의 범위에서 장<br>해 평가를 유보한다.<br>마) 장해진단 전문의는 재활의학과, 신경외과 또는<br>신경과 '
 "전문의로 한다.</p><h1 id='37' style='font-size:20px'>2) 정신행동</h1><br><p id='38' "
 "data-category='paragraph' style='font-size:20px'>가) 정신행동장해는 보험기간중에 발생한 뇌의"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001090',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
