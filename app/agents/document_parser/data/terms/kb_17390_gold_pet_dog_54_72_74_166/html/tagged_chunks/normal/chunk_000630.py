from langchain_core.documents import Document

chunk = Document(
    page_content=('. 구 분</td><td colspan="2">익자에게</td></tr><tr><td>지 급 '
 '금</td><td>액</td></tr></thead><tbody><tr><td>2대호흡계특정질환진단비</td><td>보험계약일부터 '
 '1년미만 이 특별약관의 보험가입금액 50%</td><td>보험계약일부터 1년이상 이 특별약관의 보험가입금액 '
 "100%</td></tr></tbody></table><p id='159' data-category='paragraph' "
 "style='font-size:14px'>제2조(보험금 지급에 관한"),
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
 'indexing': {'chunk_id': 'chunk_000630',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
