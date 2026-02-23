from langchain_core.documents import Document

chunk = Document(
    page_content=('- 예) 비운전자에서 운전자로 변경, 운전자에서 비운전\n'
 '- 자로 변경 등\n'
 '- ④ 이륜자동차 또는 원동기장치 자전거(전동킥보드, 전동\n'
 '- 이륜평행차, 전동기의 동력만으로 움직일 수 있는 자\n'
 '- 전거 등 개인형 이동장치를 포함)를 계속적으로 사용\n'
 '- (직업, 직무 또는 동호회 활동과 출퇴근용도 등으로\n'
 '- 주로 사용하는 경우에 한함)하게 된 경우(다만, 전동\n'
 '- 휠체어, 의료용 스쿠터 등 보행보조용 의자차는 제외\n'
 '- 합니다.)\n'
 '\uf000 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000047',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
