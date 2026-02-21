from langchain_core.documents import Document

chunk = Document(
    page_content=('부 및 설명 의무 등)를 준용하여 회사가 정한 절차에 따라 계약자는 기존 계약에\n'
 '이어 재가입할 수 있으며, 이 경우 회사는 기존계약의 가입 이후 발생한 상해 또\n'
 '는 질병을 사유로 가입을 거절할 수 없습니다.\n'
 '1. 재가입일에 있어서 반려동물의 나이가 회사가 최초가입 당시 정한 재가입 나이- 의 범위 내일 것\n'
 '- 2. 재가입 전 계약의 보험료가 정상적으로 납입완료 되었을 것\n'
 '- \uf000 이 계약의 보험기간 종료 후 계약자가 재가입을 원하는 경우 계약자는 재가입 시'),
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
 'indexing': {'chunk_id': 'chunk_000533',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
