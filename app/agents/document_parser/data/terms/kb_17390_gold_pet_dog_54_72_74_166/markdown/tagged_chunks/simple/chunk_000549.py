from langchain_core.documents import Document

chunk = Document(
    page_content=('– 3만원) x 70%, 15만원} 중 적은 금액 = 15만원 예시② 입/통원 중 수술을 한 날의 경우 ·피보험자가 부담한 수술 당일 '
 '의료비 : 203만원 ·지급금액 = {(203만원 – 3만원) x 70%, 250만원} 중 적은 금액 = 140만원 반려동물이 제1항의 '
 '치료를 받던 중에 보험기간이 180일 의료비는 보상하여 이내의 제1항의 반려동물의료비보장개시일은 계약일로부터 그날을 포함하여 상해를 '
 '직접적인 제1회 불구하고 |'),
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
 'indexing': {'chunk_id': 'chunk_000549',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
