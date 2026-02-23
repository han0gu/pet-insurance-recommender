from langchain_core.documents import Document

chunk = Document(
    page_content=("id='76' data-category='list' style='font-size:14px'>\uf000 제1항에도 불구하고 창상봉합술 "
 '치료비는 연간 3회를 한도로 하며, 한도 산정 기<br>준일자는 치료개시일(해당 상병의 진료를 위하여 최초로 내원(입원을 '
 '포함합니<br>다)한 날을 말합니다)로 합니다.<br>\uf000 제2항에서 "연간"이란 계약일로부터 매1년 단위로 도래하는 계약해당일 '
 "전까지<br>기간을 의미합니다.</p><h1 id='77' style='font-size:14px'>제2조(보험금 지급에 "
 "관한</h1><br><p id='78'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000580',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
