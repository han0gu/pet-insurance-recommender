from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 경우를 말하며 해당 산정 기준일자는 치료개시일(해당 상병의 진료를 위하여 최\n'
 '- 초로 내원(입원을 포함합니다)한 날을 말합니다)로 합니다.\n'
 '- \uf000 제1항에도 불구하고, 보건복지부에서 고시하는「건강보험 행위 급여․비급여 목록\n'
 '- 및 급여 상대가치점수」의 개정에 따라 "창상봉합술 대상 수가코드"가 폐지 또는\n'
 '86 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- 변경되어 보험금 지급사유에 대해 판정이 불가능한 경우 폐지 또는 변경 직전의 '
 '관\n'
 '- 련 법령에서 정한 "창상봉합술 대상 수가코드"를 따릅니다.'),
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
 'indexing': {'chunk_id': 'chunk_000360',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
