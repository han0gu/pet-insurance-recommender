from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제2조(용어의 정의) 이 계약에서 사용되는 용어의 정의는, 이 계약의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다. 1. '
 '계약관계 관련 용어</td></tr><tr><td>용 어</td><td>정 의</td></tr><tr><td>계약자</td><td>회사와 '
 '계약을 체결하고 보험료를 납입할 의무를 지는 사 람을 말합니다.</td></tr><tr><td>보험수익자</td><td>보험금 지급사유가 '
 '발생하는 때에 회사에 보험금을 청구하 여 받을 수 있는 사람을 말합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000002',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
