from langchain_core.documents import Document

chunk = Document(
    page_content=('- 우를 제외하고는 소를 제기하지 않습니다.\n'
 '# 제32조(관할법원)이 계약에 관한 소송 및 민사조정은 계약자의 주소지를 관할하는 법원으로 합니다. 다만, 회사와 계약\n'
 '자가 합의하여 관할법원을 달리 정할 수 있습니다.- 18 -당신에게 좋은보험 삼성화재# 제33조(소멸시효)보험금청구권, 보험료 또는 '
 '환급금 반환청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다.【소멸시효】 일정기간 행사하지 않으면 권리를 소멸시키는 제도입니다. '
 '소멸시효는 권리를 행사할 수 있는'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000085',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
