from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의 다른 조항에서 달리 정의# 같습니다.# 되지 않는 한 다음과# 1. 계약관계 관련 '
 '용어| 용 | 어 정 의 |\n'
 '| --- | --- |\n'
 '|  | 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 계약자 말합니다. |\n'
 '|  | 보험금 지급사유가 발생하는 때에 회사에 보험금을 청구하여 받 보험수익자 을 수 있는 사람을 말합니다. |\n'
 '|  | 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 보험증권 드리는 증서를 말합니다. |'),
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
 'indexing': {'chunk_id': 'chunk_000442',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
