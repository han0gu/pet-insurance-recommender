from langchain_core.documents import Document

chunk = Document(
    page_content=("포함한다.</p><br><p id='8' data-category='paragraph' style='font-size:20px'>※ "
 '주1) 안전수동 : 물체를 감별할 정도의 시력상태<br>가 아니며 눈앞에서 손의 움직임을 식별할<br>수 있을 정도의 '
 '시력상태<br>주2) 안전수지 : 시표의 가장 큰 글씨를 읽을 수<br>있는 정도의 시력은 아니나 눈 앞 30cm 이내<br>에서 '
 "손가락의 개수를 식별할 수 있을 정도<br>의 시력상태</p><br><p id='9' data-category='list'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000923',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
