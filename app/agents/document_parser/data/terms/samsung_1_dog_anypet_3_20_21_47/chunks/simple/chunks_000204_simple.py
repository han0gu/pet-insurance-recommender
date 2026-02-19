from langchain_core.documents import Document

chunk = Document(
    page_content=('- 매주 □, 매월 □, 기타 □( )\n'
 '제4조(보험료 정산기간)\n'
 '계약자는 다음 중 어느 하나의 것으로 보험료를 정산하기로 약정하고, 이 기간을 보험료정산기간 (이 하 「정산기간」 이라 합니다)이라 '
 '합니다.\n'
 '1. 계약 기간 중\n'
 '- 매월 □, - 매6개월 □, - 기타 □ ()\n'
 '2. 보험기간 종료 후 □\n'
 '제5조(예치보험료)\n'
 '계약자는 제4조(보험료 정산기간)의 매 정산기간이 시작될 때 마다 정산기간 동안의 예상 피보험자 수에 정해진 보험요율을 적용하여 산출한 '
 '보험료(이하 「예치보험료」 라 합니다)를 회사에 납입하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 41},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000204',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
