from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '용 어 풀 이 타인을 위한 계약\n'
 '계약자가 타인의 이익을 위하여 자기의 이름으로 체결하는 보험계약을 말합니다.제24조(특별약관의 해지)- \uf000 계약자는 특별약관이 '
 '소멸하기 전에는 언제든지 특별약관을 해지할 수 있습니다.\n'
 '- 다만, 타인을 위한 계약의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권\n'
 '- 을 소지한 경우에 한하여 특별약관을 해지 할 수 있습니다.\n'
 '- 상\n'
 '- \uf000 회사는 계약자 또는 피보험자의 고의로 손해가 발생한 경우 이 특별약관을 해지\n'
 '- 해\n'
 '- 할 수 있습니다.\n'
 '- 및'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000713',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
