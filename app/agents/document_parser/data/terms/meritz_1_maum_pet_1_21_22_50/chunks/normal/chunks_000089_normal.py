from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 유효한 계약으로서 그 보험종목의 변경을 요청할 때에는 회사의 '
 '사업방법서에서 정하는 방법에 따라 이를 변 경하여 드립니다. ④ 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감액하고자 할 때에는 '
 '그 감액된 부분은 해지된 것으로 보며, 이로써 회사가 환급하여야 할 보험료가 있을 경우에는 제 33조(보험료의 환급)에 따른 보험료를 '
 '계약자에게 지급합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000089',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
