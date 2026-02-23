from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한도액에 따라 보상하여 드립니다. 단, "특정처치(이물제거)"로 인한 주요치료보\n'
 '- 험금은 "이물제거(내시경)" 횟수와 "이물제거(구토유도약물)" 횟수를 합산하여\n'
 '- 연간 2회를 한도로 합니다.\n'
 '112 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)예 시 주요치료보험금의 계산\n'
 '[주요치료보험금 산출방식]\n'
 '{(피보험자가 부담한 당일 의료비 - 반려동물의료비보험금 - 자기부담금)# X 보상비율}과 치료구분별 보상한도액 중 적은 금액- '
 '[MRI/CT 시행시 지급금액 예시]'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000576',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
