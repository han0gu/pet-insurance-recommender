from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- |\n'
 '| 제1조(목적) 이 특별약관은 보험계약자(이하 "계약자"라 합니다)와 보험회사(이하 "회사"라 합니 다) 사이에 피보험자가 법률상의 '
 '배상책임을 부담함으로써 입은 손해에 대한 위험을 |\n'
 '보장하기 위하여 체결됩니다.# 제2조(용어의정의)이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의 다른 조항에서 달리 정의\n'
 '되지 않는 한 다음과 같습니다.# 1. 계약관계관련 용어| 용 어 | 정 의 |\n'
 '| --- | --- |\n'
 '| 계약자 | 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사 람을 말합니다. |'),
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
 'indexing': {'chunk_id': 'chunk_000657',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
