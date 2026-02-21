from langchain_core.documents import Document

chunk = Document(
    page_content=('“흉복부장기 또는 비뇨생식기 기능에 약간의 장해를<br>남긴 때”라 함은 아래의 경우 중 하나에 해당하는<br>때를 '
 "말한다.</p><br><p id='28' data-category='list' style='font-size:20px'>가) 방광의 "
 '용량이 50cc 이하로 위축되었거나 요도협<br>착, 배뇨기능 상실로 영구적인 간헐적 인공요도<br>가 필요한 때<br>나) 음경의 '
 '1/2 이상이 결손되었거나 질구 협착으로<br>성생활이 불가능한 때<br>다) 폐질환 또는 폐 부분절제술 후 일상생활에서 '
 '호<br>흡곤란으로 지속적인'),
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
 'indexing': {'chunk_id': 'chunk_001079',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
