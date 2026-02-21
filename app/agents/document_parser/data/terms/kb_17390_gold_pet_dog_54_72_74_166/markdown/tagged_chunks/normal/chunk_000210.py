from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하여"개인정보 보호법","신용정보의 이용 및 보호에 관한 법률" 등 관계 법령에 정\n'
 '- 한 경우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없이 수집, 이용, 조\n'
 '- 회 또는 제공하지 않습니다. 다만, 회사는 이 계약의 체결, 유지, 보험금 지급 등\n'
 '- 을 위하여 위 관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 보험회사\n'
 '- 및 보험관련단체 등에 개인정보를 제공할 수 있습니다.\n'
 '- \uf000 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.\n'
 '| 관 련 법 | 규 개인정보보호법 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000210',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
