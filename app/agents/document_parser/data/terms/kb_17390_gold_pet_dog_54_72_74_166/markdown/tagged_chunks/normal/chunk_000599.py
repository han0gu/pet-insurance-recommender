from langchain_core.documents import Document

chunk = Document(
    page_content=('- 사용한 치료를 포함합니다.)를 말합니다. 다만, 항암 치료가 아닌 다른 질환 치료\n'
 '114 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)를 목적으로 사용되나 항암 효과가 있어 부수적으로 항암제로 쓰이는 약물(스테로\n'
 '이드제, 면역억제제, 항생제, 소염진통제 등)을 사용한 치료 및 항암약물치료 이# 외의 방법으로 시행된 항암치료(항암 방사선치료, '
 '면역치료, 줄기세포치료 등)는# 제외됩니다.# 제4조(보험금을 지급하지 않는 사유)\uf000- 회사는 아래의 사유로 인한 손해는 '
 '보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000599',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
