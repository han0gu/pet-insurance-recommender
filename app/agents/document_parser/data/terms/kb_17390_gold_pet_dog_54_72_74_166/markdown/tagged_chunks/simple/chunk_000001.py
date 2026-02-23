from langchain_core.documents import Document

chunk = Document(
    page_content=('| 제1조(목적) 이 보험계약(이하 "계약"이라 합니다)은 보험계약자(이하 "계약자"라 합니다)와 보 험회사(이하 "회사"라 합니다) '
 '사이에 피보험자의 상해에 대한 위험을 보장하기 위하 여 체결됩니다. 제2조(용어의 정의) 이 계약에서 사용되는 용어의 정의는, 이 계약의 '
 '다른 조항에서 달리 정의되지 않는 한 다음과 같습니다. 1'),
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
 'indexing': {'chunk_id': 'chunk_000001',
              'chunk_char_len': 184,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
