from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약자, 피보험자가 동일하고 보험수익자가 계약자의 법정상속인인 계약일 경우</p><br><p id='25' "
 "data-category='paragraph' style='font-size:14px'>⑤ 제3항에 따라 계약이 취소된 경우에는 회사는 "
 '이미 납입한 보험료를 계약자에게 돌려<br>드리며, 보험료를 받은 기간에 대하여 ‘보험개발원이 공시하는 보험계약대출이율’을 연<br>단위 '
 "복리로 계산한 금액을 더하여 지급합니다.</p><footer id='26' style='font-size:14px'>- 13 "
 '-</footer><h1'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000125',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
