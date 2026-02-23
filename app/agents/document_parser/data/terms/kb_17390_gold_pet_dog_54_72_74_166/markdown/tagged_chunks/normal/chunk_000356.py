from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써, 그 치료\n'
 '를 목적으로 "창상봉합술(급여)"를 받은 경우 1일 1회에 한하여 이 특별약관의| 가입금액을 창상봉합술 치료비로 | 가입금액을 창상봉합술 '
 '치료비로 | 지급합니다. |\n'
 '| --- | --- | --- |\n'
 "| 창상봉합술Ⅰ (안면/경부) | 구 분 상해로 '창상봉합술(안면/경부) 대상 수가코드'에 서 정한 '창상봉합술Ⅰ(급 | 지급금액 "
 "'창상봉합술 치료비Ⅰ (안면/경부)(1일1회한, 연간3회한, 급여)'보장 |"),
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
 'indexing': {'chunk_id': 'chunk_000356',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
