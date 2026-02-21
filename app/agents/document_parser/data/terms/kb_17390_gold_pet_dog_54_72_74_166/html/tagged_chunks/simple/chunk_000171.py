from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약을 체결할 때 계약자가 청약서에 자필서명을 하지 않은 경우(자필서명에는</p><br><p id='211' "
 "data-category='paragraph' style='font-size:14px'>도장을 찍는 날인과 전자서명법 제2조 제2호에 "
 '따른 전자서명을 포함합니다)<br>\uf000 제3항에도 불구하고 전화를 이용하여 계약을 체결하는 경우 다음의 각 호의 '
 '어느<br>공<br>하나를 충족하는 때에는 자필서명을 생략할 수 있으며, 제2항의 규정에 따른 음성<br>통</p><br><p '
 "id='212'"),
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
 'indexing': {'chunk_id': 'chunk_000171',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
