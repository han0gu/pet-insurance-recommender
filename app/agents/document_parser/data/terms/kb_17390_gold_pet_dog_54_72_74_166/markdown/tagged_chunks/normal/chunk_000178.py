from langchain_core.documents import Document

chunk = Document(
    page_content=('- 부하여 위법계약의 해지를 요구할 수 있습니다.\n'
 '- \uf000 회사는 해지요구를 받은 날부터 10일 이내에 수락여부를 계약자에 통지하여야 하\n'
 '- 며, 거절할 때에는 거절 사유를 함께 통지하여야 합니다.\n'
 '68 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- \n'
 '\uf000 계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르지 않는 경우 해당 계약을\n'
 '해지할 수 있습니다.\n'
 '\uf000 제1항 및 제3항에 따라 계약이 해지된 경우 회사는 제34조(해약환급금) 제5항에\n'
 '따른 해약환급금을 계약자에게 지급합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000178',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
