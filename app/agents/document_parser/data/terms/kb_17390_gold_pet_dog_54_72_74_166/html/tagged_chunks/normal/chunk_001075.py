from langchain_core.documents import Document

chunk = Document(
    page_content=(". 무지개다리위로금(강아지, 사망)【갱신계약】</p><br><p id='53' data-category='paragraph' "
 "style='font-size:20px'>(【갱신계약】은 자동갱신으로 운영합니다)</p><br><p id='54' "
 "data-category='list' style='font-size:16px'>제1조(보험금의 지급사유)<br>\uf000 회사는 "
 '보험증권에 기재된 반려동물이 이 특별약관의 보험기간 중 무지개다리위로<br>금의 보장개시일(이하 무지개다리위로금보장개시일이라 합니다) '
 '이후에 사망한<br>경우 이 특별약관의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001075',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
