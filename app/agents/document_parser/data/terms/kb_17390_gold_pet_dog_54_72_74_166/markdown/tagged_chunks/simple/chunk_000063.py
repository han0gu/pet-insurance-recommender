from langchain_core.documents import Document

chunk = Document(
    page_content=('- 예) 자가용에서 영업용으로 변경, 영업용에서 자가용으로 변경 등\n'
 '- 법\n'
 '- 3. 보험증권 등에 기재된 피보험자의 운전여부가 변경된 경우\n'
 '- ㆍ\n'
 '- 예) 비운전자에서 운전자로 변경, 운전자에서 비운전자로 변경 등\n'
 '- 4. 이륜자동차 또는 원동기장치 자전거(전동킥보드, 전동이륜평행차, 전동기의 규정\n'
 '- 동력만으로 움직일 수 있는 자전거 등 개인형 이동장치를 포함)를 계속적으로\n'
 '- 사용(직업, 직무 또는 동호회 활동과 출퇴근용도 등으로 주로 사용하는 경우'),
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
 'indexing': {'chunk_id': 'chunk_000063',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
