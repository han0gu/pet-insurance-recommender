from langchain_core.documents import Document

chunk = Document(
    page_content=("것으로 봅니다.</p><br><p id='100' data-category='list' style='font-size:16px'>① "
 '계약자, 피보험자 및 보험수익자가 동일한 계약의 경<br>우<br>② 계약자, 피보험자가 동일하고 보험수익자가 계약자의<br>법정상속인인 '
 "계약일 경우</p><br><p id='101' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입<br>한 보험료를 계약자에게 "
 '돌려 드리며, 보험료를 받은 기간<br>에'),
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
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
