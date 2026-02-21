from langchain_core.documents import Document

chunk = Document(
    page_content=('. "창상봉합술"이라 함은 【별표10】(창상봉<br>합술(안면/경부 외) 대상 수가코드)에서 정한 창상봉합술 대상 "수가코드"에 '
 "해당<br>특</h1><br><p id='114' data-category='list' style='font-size:16px'>하는 "
 '경우를 말하며 해당 산정 기준일자는 치료개시일(해당 상병의 진료를 위하여<br>별<br>최초로 내원(입원을 포함합니다)한 날을 '
 '말합니다)로 합니다.<br>약<br>\uf000 제1항에도 불구하고, 보건복지부에서 고시하는「건강보험 행위 급여․비급여 목록 관<br>및 '
 '급여'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000605',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
