from langchain_core.documents import Document

chunk = Document(
    page_content=('조정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외하고는 소를 제기하지 않\n'
 '습니다.제35조(관할법원)이 계약에 관한 소송 및 민사조정은 계약자의 주소지를 관할하는 법원으로 합니다. 다만,- 19 -회사와 계약자가 '
 '합의하여 관할법원을 달리 정할 수 있습니다.제36조(소멸시효)보험금청구권, 보험료 또는 환급금 반환청구권은 3년간 행사하지 않으면 '
 '소멸시효(소멸시\n'
 '효는 해당 청구권을 행사할 수 있는 때로부터 진행합니다.)가 완성됩니다.【소멸시효】주어진 권리를 행사하지 않을 때 그 권리가 없어지게 '
 '되는 기간으로 보험금 지급사유가'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
