from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 변<br>경 전 보험수익자에게 보험금을 지급한 경우 변경된 보<br>험수익자에게는 별도로 보험금을 지급하지 '
 "않습니다.</p><br><p id='103' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이</p><footer "
 "id='104' style='font-size:14px'>96</footer><p id='0' "
 "data-category='paragraph' style='font-size:16px'>상 지난 유효한 계약으로서"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000354',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
