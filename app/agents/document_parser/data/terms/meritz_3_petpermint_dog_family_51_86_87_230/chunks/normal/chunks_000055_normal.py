from langchain_core.documents import Document

chunk = Document(
    page_content=('【직업】\n'
 '1) 생계유지 등을 위하여 일정한 기간동안(예: 6개월 이 상) 계속하여 종사하는 일 2) 1)에 해당하지 않는 경우에는 개인의 사회적 '
 '신분에 따르는 위치나 자리를 말함 예) 학생, 미취학아동, 무직 등\n'
 '【직무】\n'
 '직책이나 직업상 책임을 지고 담당하여 맡은 일'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 63},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
