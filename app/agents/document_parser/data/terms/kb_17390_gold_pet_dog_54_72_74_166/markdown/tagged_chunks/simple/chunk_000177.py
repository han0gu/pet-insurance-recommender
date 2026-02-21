from langchain_core.documents import Document

chunk = Document(
    page_content=('향하여 철회할 수 있으며, 서면동의 철회로 계약이 해지되어 회사가 지급하여야 할\n'
 '해약환급금이 있을 때에는 제34조(해약환급금) 제1항에 따른 해약환급금을 계약자\n'
 '에게 지급합니다.제31조의1(위법계약의 해지)# \uf000 계약자는"금융소비자보호에 관한 법률" 제47조 및 관련규정이 정하는 바에 '
 '따라- 계약체결에 대한 회사의 법위반사항이 있는 경우 계약체결일부터 5년 이내의 범위\n'
 '- 에서 계약자가 위반사항을 안 날부터 1년 이내에 계약해지요구서에 증빙서류를 첨\n'
 '- 부하여 위법계약의 해지를 요구할 수 있습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000177',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
