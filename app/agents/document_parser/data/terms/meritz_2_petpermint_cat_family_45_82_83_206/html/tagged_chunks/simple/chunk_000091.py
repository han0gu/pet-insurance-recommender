from langchain_core.documents import Document

chunk = Document(
    page_content=('계약변경 완료" data-coord="top-left:(275,1043); bottom-right:(959,1478)" '
 "/></figure><br><p id='32' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소<br>된 경우에는 보험료를 "
 "감액하고, 이후 기간 보장을 위한 재<br>원인 계약자적립액 등의 차이로 인하여 발생한 정산금액(이</p><footer id='33'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
