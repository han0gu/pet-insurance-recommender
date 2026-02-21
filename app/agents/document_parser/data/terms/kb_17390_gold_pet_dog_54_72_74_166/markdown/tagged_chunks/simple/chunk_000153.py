from langchain_core.documents import Document

chunk = Document(
    page_content=('험료(이하 "적립보험료"라 합니다)로 구성됩니다.(이하 "보장보험료"와 "적립보\n'
 '험료"를 합하여 "보험료"라 합니다)# 제26조(제2회 이후 보험료의 납입)# 계약자는 제2회 이후의 보험료를납입기일까지 납입하여야 '
 '하며, 회사는 계약자가 보 제28조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)험료를 납입한 경우에는 영수증을 발행하여 '
 '드립니다. 다만, 금융회사(우체국을 포함\n'
 '합니다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으'),
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
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
