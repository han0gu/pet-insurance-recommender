from langchain_core.documents import Document

chunk = Document(
    page_content=('- 분은 계약이 해지된 것으로 보며, 제30조(보험료의 환급)에 따라 보험료를 계약자에게 지급합니다.\n'
 '- ④ 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관을 교부하고 변경\n'
 '- 된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립니다.\n'
 '# 제20조(타인을 위한 계약)- ① 계약자는 타인을 위한 계약을 체결하는 경우에 그 타인의 위임이 없는 때에는 반드시 이를 회사에\n'
 '- 알려야 하며, 이를 알리지 않았을 때에는 그 타인은 이 계약이 체결된 사실을 알지 못하였다는 사'),
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
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
