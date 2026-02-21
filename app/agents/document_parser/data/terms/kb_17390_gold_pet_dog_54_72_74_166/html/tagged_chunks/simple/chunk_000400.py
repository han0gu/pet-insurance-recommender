from langchain_core.documents import Document

chunk = Document(
    page_content=(". 상해수술비</h1><br><p id='67' data-category='paragraph' "
 "style='font-size:14px'>제1조(보험금의 지급사유)<br>회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 "
 '직접결과로써 수술을 받은<br>경우 이 특별약관의 보험가입금액을 상해수술비로 보험수익자에게 매 사고시마다 지</p><br><h1 '
 "id='68' style='font-size:14px'>급합니다.</h1><p id='69' data-category='list' "
 "style='font-size:14px'>제2조(보험금"),
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
 'indexing': {'chunk_id': 'chunk_000400',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
