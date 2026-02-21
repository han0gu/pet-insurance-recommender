from langchain_core.documents import Document

chunk = Document(
    page_content=('비뇨생식기 기능에 심한 장해를 남<br>긴 때”라 함은 아래의 경우 중 하나에 해당하는 때를<br>말한다.</p><br><p '
 "id='21' data-category='list' style='font-size:16px'>가) 위, 대장(결장∼직장) 또는 췌장의 "
 '전부를 잘라<br>내었을 때<br>나) 소장을 3/4이상 잘라내었을 때 또는 잘라낸 소장<br>의 길이가 3m 이상일 때<br>다) '
 "간장의 3/4 이상을 잘라내었을 때<br>라) 양쪽 고환 또는 양쪽 난소를 모두 잃었을 때</p><br><p id='22'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001075',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
