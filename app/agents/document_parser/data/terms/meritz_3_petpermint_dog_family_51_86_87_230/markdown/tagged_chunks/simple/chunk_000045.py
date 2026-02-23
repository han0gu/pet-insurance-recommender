from langchain_core.documents import Document

chunk = Document(
    page_content=('무 위반을 이유로 계약을 해지하고 보험금을 지급하지\n'
 '않을 수 있습니다.# 제16조(상해보험계약 후 알릴 의무)\uf000 계약자 또는 피보험자는 보험기간 중에 피보험자에게 다\n'
 '음 각 호의 변경이 발생한 경우에는 우편, 전화, 방문 등의\n'
 '방법으로 지체없이 회사에 알려야 합니다.① 보험증권에 기재된 직업 또는 직무의 변경\n'
 '1) 현재의 직업 또는 직무가 변경된 경우\n'
 '2) 직업이 없는 자가 취직한 경우\n'
 '3) 현재의 직업을 그만둔 경우62# 【직업】- 1) 생계유지 등을 위하여 일정한 기간동안(예: 6개월 이\n'
 '- 상) 계속하여 종사하는 일'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000045',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
