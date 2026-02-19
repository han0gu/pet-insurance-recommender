from langchain_core.documents import Document

chunk = Document(
    page_content=('104 / 181\n'
 '이 특별약관이 부가된 보험계약의 경우에는 보험계약 약관의 규정에도 불구하고 다음과 같은 내용은 변경할 수 없습니다.\n'
 '1. 보험기간의 변경 2. 감액완납보험으로의 변경\n'
 '<용어풀이>\n'
 '[감액완납보험]\n'
 '차회 이후의 보험료 납입을 중단하는 대신 가입금액을 감액하는 보험\n'
 '제6조 (준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보험계약을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 105},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000680',
              'chunk_char_len': 197,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
