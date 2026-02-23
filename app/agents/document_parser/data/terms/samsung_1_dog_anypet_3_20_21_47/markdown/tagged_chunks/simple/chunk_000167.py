from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약자의 서류를 열람할 수 있습니다.\n'
 '- 41 -당신에게 좋은보험 삼성화재- ③ 회사는 제3조(피보험자의 통지)에 의해 통지된 내용에 따라 정산기간 동안의 실제보험료를 산출한 '
 '후\n'
 '- 매 정산기간 종료 후 7일 이내에 제5조(예치보험료)의 예치보험료와의 차액을 받거나 돌려드립니다.\n'
 '- 회사는 보험료가 정산되기 이전일지라도 새로이 증가 또는 교체된 피보험자에 대해 생긴 손해를\n'
 '- 보상하여 드립니다. 다만, 제3조(피보험자의 통지)의 피보험자 통지가 이루어진 경우에 한합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000167',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
