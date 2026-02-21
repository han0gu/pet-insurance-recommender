from langchain_core.documents import Document

chunk = Document(
    page_content=('는 것을 의미합니다.# 제37조(개인정보보호)- ① 회사는 이 계약과 관련된 개인정보를 이 계약의 체결, 유지, 보험금 지급 등을 위하여 '
 '「개인정보\n'
 '- 보호법」 , 「신용정보의 이용 및 보호에 관한 법률」 등 관계 법령에 정한 경우를 제외하고 계약자\n'
 '- 또는 피보험자의 동의없이 수집, 이용, 조회 또는 제공하지 않습니다. 다만, 회사는 이 계약의 체결,\n'
 '- 유지, 보험금 지급 등을 위하여 위 관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 보험\n'
 '- 회사 및 보험관련단체 등에 개인정보를 제공할 수 있습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
