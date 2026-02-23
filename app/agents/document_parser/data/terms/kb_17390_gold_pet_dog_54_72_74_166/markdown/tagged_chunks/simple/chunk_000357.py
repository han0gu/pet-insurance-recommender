from langchain_core.documents import Document

chunk = Document(
    page_content=("| 창상봉합술Ⅱ (안면/경부) | 여)(안면/경부)'을 받는 경우 상해로 '창상봉합술(안면/경부) 대상 수가코드'에 서 정한 "
 "'창상봉합술Ⅱ(급 여)(안면/경부)'을 받는 경우 | 보험가입금액 '창상봉합술 치료비Ⅱ (안면/경부)(1일1회한, 연간3회한, "
 "급여)'보장 보험가입금액 |\n"
 '- \uf000 제1항에도 불구하고 창상봉합술 치료비는 연간 3회를 한도로 하며, 한도 산정 기\n'
 '- 준일자는 치료개시일(해당 상병의 진료를 위하여 최초로 내원(입원을 포함합니\n'
 '- 다)한 날을 말합니다)로 합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000357',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
