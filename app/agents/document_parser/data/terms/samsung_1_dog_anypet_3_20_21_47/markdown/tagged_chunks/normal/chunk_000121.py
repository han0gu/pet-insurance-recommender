from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보험자를 대신하여 회사의 비용으로 이를 해결할 수 있습니다. 이 경우에 회사의 요구가 있으면 계\n'
 '- 약자 또는 피보험자는 이에 협력하여야 합니다.\n'
 '- ④ 계약자 및 피보험자가 정당한 이유 없이 제2항, 제3항의 요구에 협조하지 않았을 때에는 회사는 그\n'
 '- 로 인하여 늘어난 손해는 보상하지 않습니다.\n'
 '# 제9조(합의. 절충. 중재. 소송의 협조. 대행 등)- ① 회사는 피보험자의 법률상 손해배상책임을 확정하기 위하여 피보험자가 피해자와 '
 '행하는 합의·절'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000121',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
