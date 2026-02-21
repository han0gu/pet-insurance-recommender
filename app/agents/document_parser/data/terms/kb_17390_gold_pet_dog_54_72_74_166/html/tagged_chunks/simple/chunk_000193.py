from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 승낙을<br>서면 등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.<br>1. 보험종목<br>2. 보험기간<br>3. '
 '보험료 납입방법 및 납입기간<br>4. 계약자, 피보험자<br>5. 보험가입금액, 보험료(적립보험료를 포함합니다), 배상책임의 경우 '
 "보상한도액</p><br><p id='8' data-category='paragraph' style='font-size:14px'>등 기타 "
 '계약의 내용<br>\uf000 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 승낙이 필요하지 않<br>습니다'),
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
 'indexing': {'chunk_id': 'chunk_000193',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
