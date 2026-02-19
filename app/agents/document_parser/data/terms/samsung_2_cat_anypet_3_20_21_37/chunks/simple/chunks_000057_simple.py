from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 계약자가 제1회 보험료 등을 납입한 때부터 1년 이상 지난 유효한 계약으로서 그 보험종목 의 변경을 요청할 때에는 회사의 '
 '사업방법서에서 정하는 방법에 따라 이를 변경하여 드립니다. ③ 회사는 계약자가 제1항 제5호의 규정에 의하여 보험가입금액을 감액하고자 할 '
 '때에는 그 감액된 부 분은 계약이 해지된 것으로 보며, 제30조(보험료의 환급)에 따라 보험료를 계약자에게 지급합니다'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000057',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
