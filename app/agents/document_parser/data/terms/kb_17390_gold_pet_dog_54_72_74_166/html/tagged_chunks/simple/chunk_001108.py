from langchain_core.documents import Document

chunk = Document(
    page_content=(". 단, 계약일은 제1회 보험료를 받은 날로 합</p><br><p id='98' "
 "data-category='list'></p><br><table id='99' "
 "style='font-size:14px'><thead></thead><tbody><tr><td "
 'colspan="2">니다.</td></tr><tr><td colspan="2">예 시 반려동물장례비용지원금의 보장개시일 계약일 '
 '보장개시일 30일</td></tr><tr><td colspan="2">2024년 4월 10일 2024년 5월'),
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
 'indexing': {'chunk_id': 'chunk_001108',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
