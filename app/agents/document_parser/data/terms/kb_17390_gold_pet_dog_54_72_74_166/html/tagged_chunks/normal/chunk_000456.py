from langchain_core.documents import Document

chunk = Document(
    page_content=('보험기간 중에 상해의 직접결과로써 "외모특정상<br>및<br>해"로 진단확정되고 그 치료를 직접적인 목적으로 수술을 받은 경우 '
 '보험가입금액을<br>질<br>외모특정상해(머리,목)수술비로 보험수익자에게 매 사고시마다 지급합니다.<br>병</p><br><p '
 "id='152' data-category='paragraph' style='font-size:16px'>제2조(보험금 지급에 관한 "
 "세부규정)</p><br><p id='153' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1조(보험금의"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000456',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
