from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상) 계속하여 종사하는 일\n'
 '- 2) 1)에 해당하지 않는 경우에는 개인의 사회적 신분에\n'
 '- 따르는 위치나 자리를 말함\n'
 '- 예) 학생, 미취학아동, 무직 등\n'
 '# 【직무】직책이나 직업상 책임을 지고 담당하여 맡은 일- ② 보험증권에 기재된 피보험자의 운전 목적이 변경된 경우\n'
 '- 예) 자가용에서 영업용으로 변경, 영업용에서 자가용\n'
 '- 으로 변경 등\n'
 '- ③ 보험증권에 기재된 피보험자의 운전여부가 변경된 경우\n'
 '- 예) 비운전자에서 운전자로 변경, 운전자에서 비운전\n'
 '- 자로 변경 등'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
