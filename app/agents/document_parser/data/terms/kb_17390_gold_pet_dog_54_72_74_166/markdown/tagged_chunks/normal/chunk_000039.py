from langchain_core.documents import Document

chunk = Document(
    page_content=('- 표\n'
 '- 4. 세부적인 "보장성-1701 공시이율"의 운용방법은 회사에서 별도로 정한 "보장성\n'
 '- -1701 공시이율 적용에 관한 지침"을 따릅니다.\n'
 '- \uf000 회사는 제4항 및 제5항의 "보장성-1701 공시이율" 및 산출방법 등을 회사의 인터넷\n'
 '- 홈페이지 등에 매월 공시합니다.\n'
 '- \uf000 계약자가 제36조(중도인출)에서 정한 방법에 따라 중도인출시 인출금액 및 만기환\n'
 '- 법\n'
 '- 급금의 지급 시점까지 인출금액에 적립되었을 이자만큼 만기환급금이 감소합니다. ㆍ'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000039',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
