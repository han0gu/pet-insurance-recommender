from langchain_core.documents import Document

chunk = Document(
    page_content=('신장, 또는 간장의 장기이식을 한 경우<br>나) 장기이식을 하지 않고서는 생명유지가 불가능하<br>여 혈액투석, 복막투석 등 의료처치를 '
 "평생토록<br>받아야 할 때</p><br><h1 id='19' style='font-size:16px'>다) 방광의 저장기능과 배뇨기능을 "
 "완전히 상실한 때</h1><br><p id='20' data-category='paragraph' "
 "style='font-size:20px'>3) “흉복부장기 또는 비뇨생식기 기능에 심한 장해를 남<br>긴 때”라 함은 아래의 경우 중 "
 '하나에 해당하는'),
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
 'indexing': {'chunk_id': 'chunk_001074',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
