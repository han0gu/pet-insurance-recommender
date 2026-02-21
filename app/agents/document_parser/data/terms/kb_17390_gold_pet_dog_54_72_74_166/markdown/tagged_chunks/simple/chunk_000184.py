from langchain_core.documents import Document

chunk = Document(
    page_content=('- 율 계산"(【별표2】참조)에 따릅니다.\n'
 '- \uf000 계약자가 제36조(중도인출)에서 정한 방법에 따라 중도인출시 인출금액 및 해약환\n'
 '- 급금의 지급 시점까지 인출금액에 적립되었을 이자만큼 해약환급금이 감소합니다.\n'
 '- \uf000 회사는 경과기간별 해약환급금에 관한 표를 계약자에게 제공하여 드립니다.\n'
 '- \uf000 제31조의1(위법계약의 해지)에 따라 위법계약이 해지되는 경우 회사가 적립한 해\n'
 '지 당시의 계약자적립액 및 미경과보험료를 반환하여 드립니다.- 제35조(보험계약대출)'),
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
 'indexing': {'chunk_id': 'chunk_000184',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
