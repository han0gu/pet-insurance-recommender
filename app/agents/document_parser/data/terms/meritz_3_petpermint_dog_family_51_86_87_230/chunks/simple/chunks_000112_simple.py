from langchain_core.documents import Document

chunk = Document(
    page_content=('사실과 다른 경우에는 정정된 나이 또는 성별에 해당하는 보험금 및 보험료로 변경합니다.\n'
 '【 보험나이 계산 예시 】\n'
 '생년월일 : 1988년 10월 2일 현재(계약일) : 2023년 4월 14일\n'
 '⇒ 2023년 4월 14일 - 1988년 10월 2일\n'
 '= 34년 6월 12일 = 35세\n'
 '【 계약해당일 】\n'
 '최초계약일과 동일한 월, 일을 말합니다. 다만, 해당 연 도의 계약해당일이 없는 경우에는 해당 월의 마지막 날을 계약해당일로 합니다.\n'
 '예시1) | 계약일 : 2020년 10월 1일 -> 계약해당일 : 10월 1일'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 74},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000112',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
