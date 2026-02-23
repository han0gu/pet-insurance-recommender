from langchain_core.documents import Document

chunk = Document(
    page_content=('는 변경 전 보험수익자에게 보험금을 지급할 수 있습니다. 회사가 변경 전 보험수익자에게 보험금\n'
 '을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을 지급하지 않습니다.- ③ 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 '
 '지난 유효한 계약으로서 그\n'
 '- 보험종목의 변경을 요청할 때에는 회사의 사업방법서에서 정하는 방법에 따라 이를\n'
 '- 변경하여 드립니다.\n'
 '- ④ 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감액하고자 할 때에는 그 감액'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
