from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반려동물양육자금Ⅰ(일반상해사망)</p><br><p id='2' data-category='paragraph' "
 "style='font-size:14px'>제1조(보험금의 지급사유)<br>회사는 이 특별약관의 보험기간 중에 피보험자가 상해의 "
 "직접결과로써 사망한 경우<br>에는 이 특별약관의 보험가입금액 전액을 반려동물양육자금Ⅰ(일반상해사망)으로</p><br><p id='3' "
 "data-category='paragraph' style='font-size:14px'>보험수익자에게 지급합니다.</p><p id='4'"),
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
 'indexing': {'chunk_id': 'chunk_000351',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
