from langchain_core.documents import Document

chunk = Document(
    page_content=('② 보험증권에 기재된 피보험자의 운전 목적이 변경된 경우 예) 자가용에서 영업용으로 변경, 영업용에서 자가용 으로 변경 등 ③ 보험증권에 '
 '기재된 피보험자의 운전여부가 변경된 경우 예) 비운전자에서 운전자로 변경, 운전자에서 비운전 자로 변경 등 ④ 이륜자동차 또는 원동기장치 '
 '자전거(전동킥보드, 전동 이륜평행차, 전동기의 동력만으로 움직일 수 있는 자 전거 등 개인형 이동장치를 포함)를 계속적으로 사용 (직업, '
 '직무 또는 동호회 활동과 출퇴근용도 등으로 주로 사용하는 경우에 한함)하게 된 경우(다만, 전동 휠체어, 의료용 스쿠터 등'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 63},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
