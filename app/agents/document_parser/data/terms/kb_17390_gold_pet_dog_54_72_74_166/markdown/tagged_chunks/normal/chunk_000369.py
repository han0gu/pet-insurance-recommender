from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 "|  | 상해로 '창상봉합술(안면/경부 창상봉합술Ⅰ 외) 대상 수가코드'에 (안면/경부 서 정한 '창상봉합술Ⅰ(급 외) 여)(안면/경부 "
 "외)'을 받는 | '창상봉합술 치료비Ⅰ (안면/경부 외)(1일1회한, 연간3회한, 급여)'보장 경우 보험가입금액 |"),
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
 'indexing': {'chunk_id': 'chunk_000369',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
