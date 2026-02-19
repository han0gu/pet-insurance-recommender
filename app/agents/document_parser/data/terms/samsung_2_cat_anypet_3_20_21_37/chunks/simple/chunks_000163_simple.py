from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사는 제3조(피보험자의 통지)에 의해 통지된 내용에 따라 정산기간 동안의 실제보험료를 산출한 후 매 정산기간 종료 후 7일 이내에 '
 '제5조(예치보험료)의 예치보험료와의 차액을 받거나 돌려드립니다. 회사는 보험료가 정산되기 이전일지라도 새로이 증가 또는 교체된 피보험자에 '
 '대해 생긴 손해를 보상하여 드립니다. 다만, 제3조(피보험자의 통지)의 피보험자 통지가 이루어진 경우에 한합니다.\n'
 '제7조(보험기간의 설정)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 34},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000163',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
