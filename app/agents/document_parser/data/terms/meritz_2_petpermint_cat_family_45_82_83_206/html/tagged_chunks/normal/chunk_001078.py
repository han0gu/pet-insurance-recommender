from langchain_core.documents import Document

chunk = Document(
    page_content=("경우(치료과정에서<br>일시적으로 발생하는 경우는 제외)</p><br><p id='26' data-category='list' "
 "style='font-size:16px'>마) 심장기능 이상으로 인공심박동기를 영구적으로<br>삽입한 경우<br>바) 요도괄약근 등의 "
 "기능장해로 영구적으로 인공요<br>도괄약근을 설치한 경우</p><br><p id='27' data-category='paragraph' "
 "style='font-size:20px'>5) “흉복부장기 또는 비뇨생식기 기능에 약간의 장해를<br>남긴 때”라 함은 아래의 경우 중"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001078',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
