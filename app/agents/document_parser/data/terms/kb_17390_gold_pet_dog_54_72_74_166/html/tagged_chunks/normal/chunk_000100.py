from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이륜자동차 또는 원동기장치 자전거(전동킥보드, 전동이륜평행차, 전동기의 규정<br>동력만으로 움직일 수 있는 자전거 등 개인형 '
 '이동장치를 포함)를 계속적으로<br>사용(직업, 직무 또는 동호회 활동과 출퇴근용도 등으로 주로 사용하는 경우<br>에 한함)하게 된 '
 '경우(다만, 전동휠체어, 의료용 스쿠터 등 보행보조용 의자<br>차는 제외합니다.)<br>회사는 제1항의 통지로 인하여 위험의 변동이 '
 "발생한 경우에는 제22조(계약내용의</p><br><p id='127' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
