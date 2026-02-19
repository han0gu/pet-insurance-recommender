from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 유효한 계약으로서 그 보험종목의 변경을 요청할 때에는 회사의 '
 '사업방법서에서 정하는 방법에 따라 이를 변경하여 드립니다. ③ 회사는 계약자가 제1항 제5호에 따라 보험가입금액(배상책임의 경우 '
 '보상한도액)을 감액하고자 할 때에는 그 감액된 부분은 특별약관이 해지된 것으로 보며, 이로써 회사 가 지급하여야 할 해약환급금이 있을 '
 '때에는 이 특별약관의 해약환급금을 계약자에게 지급합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000600',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
