from langchain_core.documents import Document

chunk = Document(
    page_content=('- 102 -4. 계약자, 피보험자, 반려묘\n'
 '5. 보험가입금액(배상책임의 경우 보상한도액) 등 기타 계약의 내용- ② 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 '
 '유효한 계약으로서 그\n'
 '- 보험종목의 변경을 요청할 때에는 회사의 사업방법서에서 정하는 방법에 따라 이를\n'
 '- 변경하여 드립니다.\n'
 '- ③ 회사는 계약자가 제1항 제5호에 따라 보험가입금액(배상책임의 경우 보상한도액)을\n'
 '- 감액하고자 할 때에는 그 감액된 부분은 특별약관이 해지된 것으로 보며, 이로써 회사'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000505',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
