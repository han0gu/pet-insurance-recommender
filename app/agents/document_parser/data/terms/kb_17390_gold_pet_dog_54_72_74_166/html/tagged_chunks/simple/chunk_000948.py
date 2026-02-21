from langchain_core.documents import Document

chunk = Document(
    page_content=(": 203만원</p><br><p id='137' data-category='list' style='font-size:14px'>·지급금액 "
 '= {(203만원 – 3만원) x 70%, 250만원} 중 적은 금액 = 140만원<br>\uf000 반려동물이 제1항의 사고로 치료를 '
 '받던 중에 이 특별약관의 보험기간이 만료된<br>경우에도 만료일부터 180일 이내의 의료비는 보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000948',
              'chunk_char_len': 209,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
