from langchain_core.documents import Document

chunk = Document(
    page_content=("수 없습니다.</p><br><p id='13' data-category='list' style='font-size:16px'>① "
 '재가입일에 있어서 반려동물의 나이가 회사가 최초가<br>입 당시 정한 재가입 나이의 범위 내일 것<br>② 재가입 전 계약의 보험료가 '
 "정상적으로 납입완료 되었<br>을 것</p><br><p id='14' data-category='paragraph' "
 "style='font-size:16px'>\uf000 이 재가입 적용대상 특별약관의 보험기간 종료 후 계약<br>자가 재가입을 원하는 "
 '경우 계약자는 재가입 시점에서'),
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
 'indexing': {'chunk_id': 'chunk_000364',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
