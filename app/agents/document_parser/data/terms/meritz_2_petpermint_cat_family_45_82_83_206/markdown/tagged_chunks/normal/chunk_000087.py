from langchain_core.documents import Document

chunk = Document(
    page_content=('에 보험금 지급사유가 발생한 경우 회사는 변경 전 보험\n'
 '수익자에게 보험금을 지급할 수 있습니다. 회사가 변경\n'
 '전 보험수익자에게 보험금을 지급한 경우 변경된 보험수\n'
 '익자에게는 별도로 보험금을 지급하지 않습니다.\uf000 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이\n'
 '상 지난 유효한 계약으로서 그 보험종목의 변경을 요청할\n'
 '때에는 회사의 사업방법서에서 정하는 방법에 따라 이를 변\n'
 '경하여 드립니다.\n'
 '\uf000 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감\n'
 '액하고자 할 때에는 그 감액된 부분은 해지된 것으로 보며,'),
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
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
