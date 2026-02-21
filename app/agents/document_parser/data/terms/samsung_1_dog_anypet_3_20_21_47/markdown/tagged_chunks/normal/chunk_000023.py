from langchain_core.documents import Document

chunk = Document(
    page_content=('| 【잠복고환】 | 고환이 음낭까지 내려오지 못하는 증상 |\n'
 '제6조(손해의 통지 및 조사)① 계약자 또는 피보험자는 제4조(보상하는 손해)에서 정한 사고가 생긴 것을 안 때에는 지체없이 그\n'
 '사실을 회사에 알려야 합니다.- 7 -당신에게 좋은보험 삼성화재- ② 계약자 또는 피보험자가 제1항의 통지를 게을리하여 손해가 증가된 '
 '때에는 회사는 그 증가된 손해\n'
 '- 는 보상하여 드리지 않습니다.\n'
 '- ③ 회사가 위 제1항에 대한 손해의 사실을 확인하기 어려운 경우에는 계약자 또는 피보험자에게 필요\n'
 '- 한 증거자료의 제출을 요구할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000023',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
